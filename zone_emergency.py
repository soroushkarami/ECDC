import math
import random
import argparse
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_clustering import (
    count_ch_changes,
    baseline_style_metrics,
    track_ch_tenure,
    track_cluster_births,
    track_cluster_size,
    compute_ch_tenure_histogram
)
import zone_based_clustering_intersection as zbc
from metrics_tracking import PacketCounters, EpisodeTracker, UnifiedEventLog
from net_metrics import (
    NetworkMetrics,
    MAC_HEADER_BYTES,
    FCS_BYTES,
    SECURITY_BYTES,
    WSMP_HEADER_BYTES,
    PACKET_SIZE_BYTES,
    COMM_RANGE
)
import numpy as np

# ----------------------------------------------------------------------
# Constants for overhead accounting (Zone-based specific)
# ----------------------------------------------------------------------
CTRL_HELLO_BYTES = 64
CTRL_HELLO_NEIGH_BYTES = 128
BEACON_RATE_HZ = 10
HELLO_NEIGH_INTERVAL = 1

CTRL_HELLO_TOTAL_BYTES = CTRL_HELLO_BYTES + MAC_HEADER_BYTES + FCS_BYTES + SECURITY_BYTES + WSMP_HEADER_BYTES
CTRL_HELLO_NEIGH_TOTAL_BYTES = CTRL_HELLO_NEIGH_BYTES + MAC_HEADER_BYTES + FCS_BYTES + SECURITY_BYTES + WSMP_HEADER_BYTES

# ----------------------------------------------------------------------
AMBULANCE_SPEED = 20.0
AMBULANCE_DEPART_DEFAULT = 20.0
AMBULANCE_ROUTE = "route_Ambulance"
SFR_THRESHOLD_MS = 100.0
# ----------------------------------------------------------------------

def setup_sumo_tools():
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        sys.exit("Error: SUMO_HOME not set.")
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    return sumo_home

sumo_home = setup_sumo_tools()
import traci

def get_sumo_binary(gui=False):
    name = "sumo-gui" if gui else "sumo"
    return os.path.join(sumo_home, "bin", name + (".exe" if os.name == "nt" else ""))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zone‑based mixed topology with ambulance.")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--density", type=int, default=100,
                        help="Target average number of vehicles on the road simultaneously (excluding ambulance)")
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--amb_depart", type=float, default=AMBULANCE_DEPART_DEFAULT,
                        help="Ambulance departure time in seconds")
    return parser.parse_args()

args = parse_arguments()
TARGET_DENSITY = args.density
RANDOM_SEED = args.seed
AMBULANCE_DEPART = args.amb_depart
BASE_DIR = Path(__file__).resolve().parents[0]

net_file = "network/emergency/mixed_network_1km.net.xml"
additional_files = "network/emergency/additional.xml"

sumo_binary = get_sumo_binary(gui=args.gui)
sumo_cmd_base = [
    sumo_binary,
    "--net-file", net_file,
    "--additional-files", additional_files,
    "--step-length", "1.0",
]

# ========== Vehicle generation ==========
def define_routes():
    routes = [
        ("route_HW_E", ["highway_EB", "arm_E_out"]),
        ("route_HW_W", ["arm_E_in", "highway_WB"]),
        ("route_HW_to_N", ["highway_EB", "arm_N_out"]),
        ("route_HW_to_S", ["highway_EB", "arm_S_out"]),
        ("route_N_to_E", ["arm_N_in", "arm_E_out"]),
        ("route_N_to_W", ["arm_N_in", "highway_WB"]),
        ("route_S_to_E", ["arm_S_in", "arm_E_out"]),
        ("route_S_to_W", ["arm_S_in", "highway_WB"]),
        ("route_E_to_N", ["arm_E_in", "arm_N_out"]),
        ("route_E_to_S", ["arm_E_in", "arm_S_out"]),
        ("route_Ambulance", ["highway_EB", "arm_E_out"]),
    ]
    for route_id, edges in routes:
        traci.route.add(route_id, edges)

def generate_vehicles(target_density, seed, amb_depart, injection_horizon=400):
    """
    Generate vehicles over [0, amb_depart + injection_horizon] such that the
    average number of vehicles on the road equals target_density.
    """
    rn = random.Random(seed)
    highway_routes = ['route_HW_E', 'route_HW_W', 'route_HW_to_N', 'route_HW_to_S']
    city_routes = ['route_N_to_E', 'route_N_to_W', 'route_S_to_E', 'route_S_to_W',
                   'route_E_to_N', 'route_E_to_S']
    highway_prob = 0.75

    avg_trip_time = 200.0

    total_injection_window = amb_depart + injection_horizon
    total_vehicles = int(math.ceil(target_density * total_injection_window / avg_trip_time))

    print(f"Generating {total_vehicles} vehicles to maintain target density of {target_density} "
          f"(avg trip time = {avg_trip_time:.0f}s, injection window = {total_injection_window:.0f}s)")

    for i in range(total_vehicles):
        veh_id = f'veh{i}'
        speed = rn.uniform(3.611111, 13.888888)
        lane = rn.choice([0, 1])
        if rn.random() < highway_prob:
            route = rn.choice(highway_routes)
        else:
            route = rn.choice(city_routes)

        depart_time = rn.uniform(0, total_injection_window)
        traci.vehicle.add(
            veh_id, route, typeID='car',
            departSpeed=speed, depart=depart_time, departLane=str(lane)
        )

# ========== Helper functions ==========
def get_vehicle_data(vehicle_id, rsu_pos):
    x, y = traci.vehicle.getPosition(vehicle_id)
    vehicle_type = traci.vehicle.getTypeID(vehicle_id)
    route_id = traci.vehicle.getRouteID(vehicle_id)
    edge_id = traci.vehicle.getRoadID(vehicle_id)
    lane_id = traci.vehicle.getLaneID(vehicle_id)
    lane_index = traci.vehicle.getLaneIndex(vehicle_id)
    lane_position = traci.vehicle.getLanePosition(vehicle_id)
    speed = traci.vehicle.getSpeed(vehicle_id)
    acceleration = traci.vehicle.getAcceleration(vehicle_id)
    allowed_speed = traci.vehicle.getAllowedSpeed(vehicle_id)
    angle = traci.vehicle.getAngle(vehicle_id)
    traveled_dist = traci.vehicle.getDistance(vehicle_id)
    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
    if rsu_pos:
        min_dist_to_rsu = min(math.hypot(x-rx, y-ry) for rx, ry in rsu_pos)
    else:
        min_dist_to_rsu = -1.0
    return {
        "vehicle_id": vehicle_id, "vehicle_type": vehicle_type, "route_id": route_id,
        "edge_id": edge_id, "lane_id": lane_id, "lane_index": lane_index,
        "lane_position": lane_position, "x": x, "y": y, "speed": speed,
        "acceleration": acceleration, "allowed_speed": allowed_speed, "angle": angle,
        "traveled_dist": traveled_dist, "waiting_time": waiting_time,
        "min_dist_to_rsu": min_dist_to_rsu
    }

def update_clustering_manager_state(manager, vehicle_ids, rsu_positions):
    current_ids = set(vehicle_ids)
    for veh_id in set(manager.vehicles.keys()) - current_ids:
        del manager.vehicles[veh_id]
    for veh_id in vehicle_ids:
        data = get_vehicle_data(veh_id, rsu_positions)
        rad = math.radians(data['angle'])
        dest = zbc.Position(data['x'] + 1000 * math.cos(rad),
                            data['y'] + 1000 * math.sin(rad))
        if veh_id in manager.vehicles:
            v = manager.vehicles[veh_id]
            v.position = zbc.Position(data['x'], data['y'])
            v.speed = data['speed']
            v.lane_id = data['lane_id']
            v.edge_id = data['edge_id']
            v.angle = data['angle']
            v.destination = dest
            v.update_direction()
        else:
            v = zbc.VehicleState(
                veh_id=veh_id,
                lane_id=data['lane_id'],
                edge_id=data['edge_id'],
                position=zbc.Position(data['x'], data['y']),
                speed=data['speed'],
                destination=dest
            )
            v.angle = data['angle']
            v.update_direction()
            manager.add_vehicle(v)

def run_clustering(manager, time_step):
    zone_ch_dict = manager.run()
    zone_ch = {zone_id: info for zone_id, info in zone_ch_dict.items()}
    print(f'\n=== Clustering at t={time_step:.1f} s ===')
    for zone_id, ch_id in zone_ch.items():
        for rsu in manager.rsus.values():
            if zone_id in rsu.zones:
                zone = rsu.zones[zone_id]
                member_count = len(zone.active_members) + len(zone.dead_members)
                print(f'    Zone {zone_id}: CH={ch_id}, members={member_count}')
                break
    return zone_ch

def get_zone_members(manager):
    members = {}
    for rsu in manager.rsus.values():
        for zone in rsu.zones.values():
            zone_members = {
                vid for vid in (zone.active_members | zone.dead_members)
                if vid in manager.vehicles and manager.vehicles[vid].mode in (zbc.VehicleMode.CH, zbc.VehicleMode.CM)
            }
            if len(zone_members) >= 3:
                members[zone.zone_id] = zone_members
    return members

# ========== Zone configuration ==========
def configure_highway_zones_intersection_style(rsu, lane_y_coords, coverage_radius, lane_start, lane_end):
    rsu_x = rsu.position.x
    coverage_start = rsu_x - coverage_radius
    coverage_end = rsu_x + coverage_radius
    segment_start = max(lane_start, coverage_start)
    segment_end = min(lane_end, coverage_end)
    if segment_start >= segment_end:
        return
    for lane_id, y_coord in lane_y_coords.items():
        start_pos = zbc.Position(segment_start, y_coord)
        end_pos = zbc.Position(segment_end, y_coord)
        rsu.set_lane_geometry(lane_id, start_pos, end_pos)
        rsu.configure_lane_zones(lane_id)

# ========== Emergency tracker ==========
class EmergencyTracker:
    def __init__(self):
        self.amb_positions = {}  # time_step -> (x, y) of ambulance for dynamic EICD

        # EICD – position‑based
        self.zone_entries = []          # (zone_id, entry_time, amb_pos)
        self.change_events = []         # (time, ch_veh_id, ch_pos)

        # EMDD / E2ED
        self.emergency_delays = []

        # Ambulance PDR / STP
        self.ambulance_packets_sent = 0
        self.ambulance_packets_received = 0

        # SFR
        self.sfr_fail_count = 0
        self.sfr_total = 0
        self.sfr_threshold = SFR_THRESHOLD_MS / 1000.0
        self.n_loss = 0
        self.n_late = 0
        self.n_dissolutions = 0

        # Emergency Throughput
        self.ambulance_successful_packets = 0

        # Ambulance transit & speed
        self.amb_departure_time = None
        self.amb_arrival_time = None
        self.amb_speeds = []

        # Neighbour / coverage
        self.neighbor_counts = []
        self.coverage_fractions = []
        self.cluster_connectivity_ratios = []   # CCR

        # Time-series data
        self.eicd_cumulative_over_time = []  # list of (time, cumulative_eicd_count_so_far)
        self.ccr_over_time = []  # list of (time, ccr_value)
        self.amb_packet_history = []  # list of (time, success_flag) for sliding window
        self.apdr_over_time = []  # list of (time, sliding_apdr)

    def record_ch_change(self, time_step, ch_vehicle_id, ch_position, kind='transfer'):
        self.change_events.append((time_step, ch_vehicle_id, ch_position))
        if kind == 'dissolution':
            self.n_dissolutions += 1

    def compute_eicd(self):
        """Compute Emergency-Induced Cluster Disruption using a dynamic sliding window.
        Counts unique CH changes that occur within 250m of the ambulance's CURRENT position."""
        unique_events = set()
        for event_time, _, event_pos in self.change_events:
            amb_pos_at_event = self.amb_positions.get(event_time)
            if amb_pos_at_event is None:
                continue
            if math.hypot(event_pos[0] - amb_pos_at_event[0],
                          event_pos[1] - amb_pos_at_event[1]) <= 250.0:
                unique_events.add((event_time, event_pos))
        return len(unique_events)

    def get_cumulative_eicd_up_to(self, time_step):
        count = 0
        for event_t, _, event_p in self.change_events:
            if event_t <= time_step:
                amb_pos = self.amb_positions.get(event_t)
                if amb_pos and (math.hypot(event_p[0] - amb_pos[0], event_p[1] - amb_pos[1]) <= 250.0):
                    count += 1
        return count

    def record_emergency_packet(self, delay, success, neighbors_within_range, total_vehicles_excl_amb):
        self.ambulance_packets_sent += 1
        self.sfr_total += 1
        if success:
            self.ambulance_packets_received += 1
            self.ambulance_successful_packets += 1
            self.emergency_delays.append(delay)
            if delay > self.sfr_threshold:
                self.sfr_fail_count += 1
                self.n_late += 1
        else:
            self.sfr_fail_count += 1
            self.n_loss += 1
        self.neighbor_counts.append(neighbors_within_range)
        if total_vehicles_excl_amb > 0:
            self.coverage_fractions.append(neighbors_within_range / total_vehicles_excl_amb)
        else:
            self.coverage_fractions.append(0.0)

    def record_ambulance_speed(self, speed):
        self.amb_speeds.append(speed)

    def record_ambulance_pos(self, timestep, pos):
        self.amb_positions[timestep] = pos

    def set_departure(self, time_step):
        self.amb_departure_time = time_step

    def set_arrival(self, time_step):
        self.amb_arrival_time = time_step

    def get_transit_time(self):
        if self.amb_departure_time is not None and self.amb_arrival_time is not None:
            return self.amb_arrival_time - self.amb_departure_time
        return 0.0

    def get_avg_speed(self):
        if self.amb_speeds:
            return sum(self.amb_speeds) / len(self.amb_speeds)
        return 0.0

    def get_emergency_throughput(self):
        transit = self.get_transit_time()
        if transit > 0:
            return self.ambulance_successful_packets / transit
        return 0.0

    def get_sfr(self):
        if self.sfr_total > 0:
            return self.sfr_fail_count / self.sfr_total
        return 0.0

    def get_stp(self):
        if self.ambulance_packets_sent > 0:
            return self.ambulance_packets_received / self.ambulance_packets_sent
        return 0.0

    def get_e2ed_stats(self):
        if self.emergency_delays:
            return sum(self.emergency_delays)/len(self.emergency_delays), max(self.emergency_delays), min(self.emergency_delays)
        return 0.0, 0.0, 0.0

    def get_neighbor_avg(self):
        if self.neighbor_counts:
            return sum(self.neighbor_counts) / len(self.neighbor_counts)
        return 0.0

    # ---- CCR methods ----
    def record_ambulance_connectivity(self, ambulance_zone, curr_zone_members_dict, veh_pos, comm_range):
        if ambulance_zone is None:
            return
        members = curr_zone_members_dict.get(ambulance_zone, set())
        if "ambulance" not in members:
            return
        total_members = len(members) - 1
        if total_members == 0:
            self.cluster_connectivity_ratios.append(0.0)
            return
        amb_pos = veh_pos["ambulance"]
        connected = 0
        for vid in members:
            if vid == "ambulance":
                continue
            dist = math.hypot(amb_pos[0] - veh_pos[vid][0],
                              amb_pos[1] - veh_pos[vid][1])
            if dist <= comm_range:
                connected += 1
        ratio = connected / total_members
        self.cluster_connectivity_ratios.append(ratio)

    def get_ccr_avg(self):
        if self.cluster_connectivity_ratios:
            return sum(self.cluster_connectivity_ratios) / len(self.cluster_connectivity_ratios)
        return 0.0

    def get_ccr_max(self):
        if self.cluster_connectivity_ratios:
            return max(self.cluster_connectivity_ratios)
        return 0.0

    def get_ccr_min(self):
        if self.cluster_connectivity_ratios:
            return min(self.cluster_connectivity_ratios)
        return 0.0

# ========== Main ==========
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    import time as _time
    _step_times = []
    clustering_dir = os.path.join(base_dir, "metrics", "zone_based_emergency", "clustering_data")
    network_dir = os.path.join(base_dir, "metrics", "zone_based_emergency", "network_data")
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)
    RSU_COVERAGE = float(os.environ.get('ECDC_RSU_COVERAGE', str(COMM_RANGE)))
    eta_relay = float(os.environ.get('ECDC_ETA_RELAY', '0.15'))
    gamma = float(os.environ.get('ECDC_GAMMA', '0.50'))
    load_factor = float(os.environ.get('ECDC_LOAD_FACTOR', '1.0'))

    amb_depart_int = int(AMBULANCE_DEPART)
    RUN_TAG = os.environ.get('ECDC_RUN_TAG', '').strip()
    TAG_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
    _file_suffix = TAG_SUFFIX
    agg_csv = os.path.join(clustering_dir,
                           f"zone_emergency{args.density}_5rsu_depart{amb_depart_int}{_file_suffix}.csv")
    if not os.path.exists(agg_csv):
        with open(agg_csv, 'w') as f:
            f.write(
                "density,rep,seed,avg_ch_lifetime,total_ch_changes,total_cl_changes,avg_ch_changes,"
                "avg_cluster_changes,avg_cluster_lifetime,pdr,throughput_bps,overhead_bps,"
                "energy_bits_per_joule,avg_delay,jitter,"
                "total_clusters_created,avg_cluster_size,avg_ch_tenure,median_ch_tenure,max_ch_tenure,"
                "emdd_avg,emdd_max,emdd_min,eicd_total,ambulance_pdr,"
                "sfr,emergency_throughput,amb_transit_time,amb_avg_speed,amb_avg_neighbors,ccr_avg,ccr_max,ccr_min,"
                "coverage_stability,avg_ch_assignment_duration,link_stability_ratio,avg_link_duration,n_loss,n_late,n_dissolutions,"
                "rch_util_pct,gate_v2v_fail_pct,gate_v2i_fail_pct,relay_succ_pct,direct_succ_pct,"
                "relay_mean_delay_ms,direct_mean_delay_ms,rch_none_pct,rch_avail_count,"
                                "network_pdr,emergency_sent,ordinary_sent,disconnected_seconds,total_ambulance_seconds,"
                "relay_delivery_pct,direct_delivery_pct,n_births,n_transfers\n"
            )

    for rep in range(args.reps):
        current_seed = RANDOM_SEED + rep

        random.seed(current_seed)

        np.random.seed(current_seed)
        rng_pkt = random.Random(current_seed * 1000 + 1)
        sumo_cmd = sumo_cmd_base + ["--seed", str(current_seed)]
        if args.gui:
            sumo_cmd.extend(["--delay", "200", "--start"])

        traci.start(sumo_cmd)

        traci.vehicletype.copy("DEFAULT_VEHTYPE", "car")
        define_routes()
        generate_vehicles(args.density, current_seed, AMBULANCE_DEPART, injection_horizon=400)

        # ---- Add ambulance ----
        if "emergency" not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("car", "emergency")
            traci.vehicletype.setMaxSpeed("emergency", 25.0)
            traci.vehicletype.setColor("emergency", (255, 0, 0))
            traci.vehicletype.setAccel("emergency", 2.6)
            traci.vehicletype.setDecel("emergency", 4.5)
            traci.vehicletype.setParameter("emergency", "sigma", "0.5")
            traci.vehicletype.setMinGap("emergency", 2.5)

        traci.vehicle.add(
            vehID="ambulance",
            routeID="route_Ambulance",
            typeID="emergency",
            depart=AMBULANCE_DEPART,
            departSpeed=AMBULANCE_SPEED,
            departLane="0"
        )

        bs_pos = None
        rsu_positions = []
        for poi_id in traci.poi.getIDList():
            if traci.poi.getType(poi_id) == "base_station":
                bs_pos = traci.poi.getPosition(poi_id)
            elif traci.poi.getType(poi_id) == "RSU":
                rsu_positions.append(traci.poi.getPosition(poi_id))

        if bs_pos is None:
            bs_pos = (1250.0, 100.0)

        from net_metrics import set_channel_seed
        set_channel_seed(current_seed * 1000 + 2)
        net_metrics = NetworkMetrics(sim_time_total=0.0)
        emergency_tracker = EmergencyTracker()
        pkt_counters = PacketCounters()
        event_log = UnifiedEventLog()
        link_tracker = EpisodeTracker(COMM_RANGE, close_on_range=True)
        ch_tenure_tr = EpisodeTracker(COMM_RANGE, close_on_range=False)
        assign_tr = EpisodeTracker(COMM_RANGE, close_on_range=False)

        clustering_manager = zbc.ClusteringManager(
            dsrc_v2v_range=COMM_RANGE,
            delta_t=1.0,
            vehicle_length_l=5.0,
            A=0.5,
            B=0.5
        )

        # ---- Highway RSUs ----
        #highway_rsu_positions = [(250, 12), (750, 12), (1250, 12), (1750, 12)]
        highway_rsu_positions = [(200, 12), (600, 12)]
        lane_y_coords = {
            "highway_EB_0": -4.8,
            "highway_EB_1": -1.6,
            "highway_WB_0": 4.8,
            "highway_WB_1": 1.6,
        }
        for idx, (x, y) in enumerate(highway_rsu_positions):
            rsu = zbc.RSU(rsu_id=f'rsu_{idx}', position=zbc.Position(x, y), coverage_range=RSU_COVERAGE)
            clustering_manager.add_rsu(rsu)
            configure_highway_zones_intersection_style(
                rsu, lane_y_coords, coverage_radius=RSU_COVERAGE,
                lane_start=0.0, lane_end=1000.0
            )

        # ---- Intersection RSU ----
        rsu_intersection = zbc.RSU(rsu_id='rsu_4', position=zbc.Position(1000.0, 12.0), coverage_range=RSU_COVERAGE)
        clustering_manager.add_rsu(rsu_intersection)
        for lane_id in traci.lane.getIDList():
            if lane_id.startswith('arm_'):
                shape = traci.lane.getShape(lane_id)
                if len(shape) >= 2:
                    start_pos = zbc.Position(shape[0][0], shape[0][1])
                    end_pos = zbc.Position(shape[-1][0], shape[-1][1])
                    rsu_intersection.set_lane_geometry(lane_id, start_pos, end_pos)
                    rsu_intersection.configure_lane_zones(lane_id)

        # Metrics containers
        prev_zone_ch_dict = {}
        prev_zone_members_dict = {}
        membership_birth_time = {}
        membership_lifetimes = []
        total_ch_changes = 0
        total_cl_changes = 0
        steps = []

        covered_vehicle_sum = 0.0
        total_vehicle_steps = 0

        link_up_steps = 0

        total_vehicle_seconds = 0.0

        birth_counter = {}
        size_list = []
        ch_start_time = {}
        ch_end_time = {}

        prev_ambulance_zone = None
        ambulance_departed = False
        ambulance_arrived = False

        network_log = os.path.join(network_dir,
                                   f"zone_emergency{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_5rsu{_file_suffix}.txt")
        per_step_csv = os.path.join(clustering_dir,
                                    f"per_step_zone_emergency{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_5rsu{_file_suffix}.csv")
        with open(per_step_csv, 'w') as csv_file:
            csv_file.write(
                "time_step,ch_changes_this_step,cl_changes_this_step,cumulative_ch_changes,cumulative_cl_changes,avg_lifetime,cumulative_eicd,sliding_apdr,ccr\n")

        emergency_delays_csv = os.path.join(
            clustering_dir,
            f"emergency_delays{args.density}_seed{current_seed}_depart{amb_depart_int}{_file_suffix}.csv")
        with open(emergency_delays_csv, 'w') as edf:
            edf.write("time_step,delay_s,success\n")

        def write_header(file):
            header = (
                "time\tvehicle_id\tvehicle_type\troute_id\tedge_id\tlane_id\t"
                "lane_index\tlane_position_m\tx_m\ty_m\tspeed_m_per_s\t"
                "acceleration_m_per_s2\tallowed_speed_m_per_s\tangle_degree\t"
                "traveled_dist_m\twaiting_time\tmin_dist_rsu_m\n"
            )
            file.write(header)

        def write_vehicle_data(file, time_step, data):
            row = (
                f"{time_step:.1f}\t{data['vehicle_id']}\t{data['vehicle_type']}\t{data['route_id']}\t"
                f"{data['edge_id']}\t{data['lane_id']}\t{data['lane_index']}\t{data['lane_position']:.2f}\t"
                f"{data['x']:.2f}\t{data['y']:.2f}\t{data['speed']:.2f}\t{data['acceleration']:.2f}\t"
                f"{data['allowed_speed']:.2f}\t{data['angle']:.2f}\t{data['traveled_dist']:.2f}\t"
                f"{data['waiting_time']:.2f}\t{data['min_dist_to_rsu']:.2f}\n"
            )
            file.write(row)

        try:
            with open(network_log, 'w', encoding='utf-8') as log_file:
                write_header(log_file)

                step = 0
                while traci.simulation.getMinExpectedNumber() > 0:
                    _t0 = _time.perf_counter()
                    traci.simulationStep()
                    step += 1
                    time_step = traci.simulation.getTime()
                    vehicle_ids = sorted(traci.vehicle.getIDList())
                    total_vehicle_seconds += len(vehicle_ids) * 1.0

                    sliding_apdr = 0.0

                    amb_present = "ambulance" in vehicle_ids
                    if amb_present and not ambulance_departed:
                        emergency_tracker.set_departure(time_step)
                        ambulance_departed = True

                    update_clustering_manager_state(clustering_manager, vehicle_ids, rsu_positions)

                    for vid in vehicle_ids:
                        data = get_vehicle_data(vid, rsu_positions)
                        write_vehicle_data(log_file, time_step, data)
                    log_file.flush()

                    if len(vehicle_ids) > 1:
                        curr_zone_ch_dict = run_clustering(clustering_manager, time_step)
                        curr_zone_members_dict = get_zone_members(clustering_manager)

                        if amb_present:
                            amb_veh = clustering_manager.vehicles.get("ambulance")
                            ambulance_zone = amb_veh.zone_id if amb_veh else None

                            if ambulance_zone is None:
                                # Find zone by position
                                amb_pos = traci.vehicle.getPosition("ambulance")
                                amb_lane = traci.vehicle.getLaneID("ambulance")
                                for rsu in clustering_manager.rsus.values():
                                    for zone in rsu.zones.values():
                                        if hasattr(zone, 'contains_x'):
                                            if zone.lane_id == amb_lane and zone.contains_x(amb_pos[0]):
                                                ambulance_zone = zone.zone_id
                                                break
                                        else:
                                            # circular zones not used in this version
                                            pass
                                    if ambulance_zone:
                                        break

                            if ambulance_zone is not None:
                                # Find owner RSU
                                owner_rsu = None
                                for rsu in clustering_manager.rsus.values():
                                    if ambulance_zone in rsu.zones:
                                        owner_rsu = rsu
                                        break

                                if amb_veh:
                                    amb_veh.zone_id = ambulance_zone
                                    amb_veh.rsu_id = owner_rsu.rsu_id if owner_rsu else None

                                # Add to members if missing
                                if ambulance_zone not in curr_zone_members_dict:
                                    curr_zone_members_dict[ambulance_zone] = {"ambulance"}
                                else:
                                    curr_zone_members_dict[ambulance_zone].add("ambulance")

                                if ambulance_zone not in curr_zone_ch_dict or curr_zone_ch_dict[ambulance_zone] is None:
                                    curr_zone_ch_dict[ambulance_zone] = "ambulance"
                                    if amb_veh:
                                        amb_veh.mode = zbc.VehicleMode.CH
                                        amb_veh.connected_ch_id = "ambulance"

                        # ----- EICD: record CH changes with positions -----
                        for zone_id, new_ch in curr_zone_ch_dict.items():
                            if new_ch is None:
                                continue
                            old_ch = prev_zone_ch_dict.get(zone_id)
                            if new_ch in vehicle_ids:
                                event_log.note_ch_position(zone_id,
                                                           traci.vehicle.getPosition(new_ch))
                            if old_ch is None:
                                if new_ch in vehicle_ids:
                                    event_log.record(time_step, 'birth', zone_id,
                                                     traci.vehicle.getPosition(new_ch), ch_id=new_ch)
                            elif old_ch != new_ch:
                                if new_ch in vehicle_ids:
                                    event_log.record(time_step, 'transfer', zone_id,
                                                     traci.vehicle.getPosition(new_ch),
                                                     ch_id=new_ch, old_ch_id=old_ch)

                        for zone_id, old_ch in prev_zone_ch_dict.items():
                            if zone_id in curr_zone_ch_dict or old_ch is None:
                                continue
                            pos = event_log.last_position(zone_id)
                            if old_ch in vehicle_ids:
                                pos = traci.vehicle.getPosition(old_ch)
                            event_log.record(time_step, 'dissolution', zone_id, pos,
                                             old_ch_id=old_ch)


                        ch_changes_this = count_ch_changes(prev_zone_ch_dict, curr_zone_ch_dict)
                        membership_birth_time, membership_lifetimes, cl_changes_this = baseline_style_metrics(
                            prev_zone_members_dict, curr_zone_members_dict,
                            membership_birth_time, membership_lifetimes, time_step
                        )

                        birth_counter = track_cluster_births(curr_zone_members_dict, prev_zone_members_dict,
                                                             birth_counter, time_step)
                        size_list = track_cluster_size(curr_zone_members_dict, size_list)

                        # ---- Overhead recording (Zone-based) ----
                        net_metrics.record_control_overhead(
                            len(vehicle_ids) * CTRL_HELLO_TOTAL_BYTES * BEACON_RATE_HZ,
                            num_packets=len(vehicle_ids) * BEACON_RATE_HZ
                        )
                        hello_neigh_rate = 1.0 / HELLO_NEIGH_INTERVAL
                        net_metrics.record_control_overhead(
                            len(vehicle_ids) * CTRL_HELLO_NEIGH_TOTAL_BYTES * hello_neigh_rate,
                            num_packets=int(len(vehicle_ids) * hello_neigh_rate + 0.5)
                        )

                        total_ch_changes += ch_changes_this
                        total_cl_changes += cl_changes_this

                        # ---- Compute neighbor_counts and bs_neighbor_count BEFORE ambulance handling ----
                        veh_pos = {vid: traci.vehicle.getPosition(vid) for vid in vehicle_ids}

                        ch_tenure_tr.update(
                            time_step, vehicle_ids, veh_pos,
                            {ch: cid for cid, ch in curr_zone_ch_dict.items() if ch is not None},
                            {ch: True for ch in curr_zone_ch_dict.values() if ch is not None},
                        )
                        neighbor_counts = {}
                        for vid in vehicle_ids:
                            if vid == "ambulance":
                                continue
                            count = 0
                            pos_v = veh_pos[vid]
                            for other in vehicle_ids:
                                if other == vid or other == "ambulance":
                                    continue
                                pos_o = veh_pos[other]
                                dist = math.hypot(pos_v[0] - pos_o[0], pos_v[1] - pos_o[1])
                                if dist <= COMM_RANGE:
                                    count += 1
                            neighbor_counts[vid] = count

                        bs_neighbor_count = 0
                        bs_pos_xy = bs_pos
                        for vid in vehicle_ids:
                            if vid == "ambulance":
                                continue
                            pos_v = veh_pos[vid]
                            dist_to_bs = math.hypot(pos_v[0] - bs_pos_xy[0], pos_v[1] - bs_pos_xy[1])
                            if dist_to_bs <= COMM_RANGE:
                                bs_neighbor_count += 1

                        # ---- Ambulance handling ----
                        if amb_present:
                            amb_pos = traci.vehicle.getPosition("ambulance")
                            emergency_tracker.record_ambulance_pos(time_step, amb_pos)
                            amb_speed = traci.vehicle.getSpeed("ambulance")
                            emergency_tracker.record_ambulance_speed(amb_speed)

                            amb_veh = clustering_manager.vehicles.get("ambulance")
                            ambulance_zone = amb_veh.zone_id if amb_veh else None

                            # Record connectivity EVERY step
                            if ambulance_zone is not None:
                                emergency_tracker.record_ambulance_connectivity(
                                    ambulance_zone, curr_zone_members_dict, veh_pos, COMM_RANGE
                                )

                            # Send emergency packet every second (unconditional)
                            if int(time_step) % 1 == 0:
                                neighbors = 0
                                total_excl = len(vehicle_ids) - 1
                                amb_x, amb_y = amb_pos
                                for vid in vehicle_ids:
                                    if vid == 'ambulance':
                                        continue
                                    vx, vy = veh_pos[vid]
                                    if math.hypot(vx - amb_x, vy - amb_y) < COMM_RANGE:
                                        neighbors += 1
                                pkt_counters.emergency_sent += 1
                                pkt_counters.total_ambulance_seconds += 1
                                ch = None
                                if ambulance_zone is not None:
                                    ch = curr_zone_ch_dict.get(ambulance_zone)
                                if ch is not None and ch != "ambulance":
                                    # 2-hop: ambulance -> CH (V2V) -> BS (V2I)
                                    attempted_hops = 2
                                    num_contenders_v2v = neighbor_counts.get(ch, 0)
                                    num_contenders_v2i = bs_neighbor_count
                                else:
                                    # 1-hop: direct V2I (ambulance is CH of forced singleton, or unclustered)
                                    attempted_hops = 1
                                    num_contenders_v2v = None
                                    num_contenders_v2i = bs_neighbor_count
                                pkt_counters.emergency_hops += attempted_hops
                                pkt_counters.emergency_tx_bytes += attempted_hops * (
                                        PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES
                                        + SECURITY_BYTES + WSMP_HEADER_BYTES)
                                if ch is not None and ch != "ambulance":
                                    success, delay, _ = net_metrics.simulate_application_report(
                                        "ambulance", ch, bs_pos, veh_pos, veh_pos,
                                        num_contenders_v2v=num_contenders_v2v,
                                        num_contenders_v2i=num_contenders_v2i,
                                        load_factor=load_factor,
                                        is_emergency=True,
                                    )
                                else:
                                    success, delay, _ = net_metrics.simulate_application_report(
                                        "ambulance", "ambulance", bs_pos, veh_pos, veh_pos,
                                        num_contenders=num_contenders_v2i,
                                        load_factor=load_factor,
                                        is_emergency=True,
                                    )
                                if success:
                                    pkt_counters.emergency_recv += 1
                                    if delay > SFR_THRESHOLD_MS / 1000.0:
                                        pkt_counters.emergency_late += 1
                                emergency_tracker.record_emergency_packet(delay, success, neighbors, total_excl)
                                with open(emergency_delays_csv, 'a') as edf:
                                    edf.write(f"{time_step},{delay},{1 if success else 0}\n")
                                emergency_tracker.amb_packet_history.append((time_step, success))
                                cutoff_time = time_step - 5
                                emergency_tracker.amb_packet_history = [
                                    (t, s) for t, s in emergency_tracker.amb_packet_history
                                    if t >= cutoff_time]
                                if emergency_tracker.amb_packet_history:
                                    sliding_success = sum(1 for _, s in emergency_tracker.amb_packet_history if s)
                                    sliding_total = len(emergency_tracker.amb_packet_history)
                                    sliding_apdr = sliding_success / sliding_total
                                else:
                                    sliding_apdr = 0

                        # ---- Application traffic for normal vehicles ----
                        if len(vehicle_ids) > 2:
                            veh_to_ch = {}
                            for zone_id, members in curr_zone_members_dict.items():
                                ch = curr_zone_ch_dict.get(zone_id)
                                if ch:
                                    for m in members:
                                        veh_to_ch[m] = ch
                            for ch in curr_zone_ch_dict.values():
                                if ch:
                                    veh_to_ch[ch] = ch

                            for _ in range(20):
                                src = rng_pkt.choice(vehicle_ids)
                                if src == "ambulance":
                                    continue
                                ch = veh_to_ch.get(src)
                                net_metrics.packets_sent += 1
                                pkt_counters.ordinary_sent += 1
                                if ch:
                                    hops = 1 if src == ch else 2
                                    net_metrics.record_data_hops(hops)
                                    pkt_counters.ordinary_hops += hops
                                    pkt_counters.ordinary_tx_bytes += hops * (
                                            PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES
                                            + SECURITY_BYTES + WSMP_HEADER_BYTES)

                                    if src == ch:
                                        num_contenders_v2i = bs_neighbor_count
                                        success, delay, _ = net_metrics.simulate_application_report(
                                            src, ch, bs_pos, veh_pos, veh_pos,
                                            num_contenders=num_contenders_v2i,
                                            load_factor=load_factor
                                        )
                                    else:
                                        num_contenders_v2v = neighbor_counts.get(ch, 0)
                                        num_contenders_v2i = bs_neighbor_count
                                        success, delay, _ = net_metrics.simulate_application_report(
                                            src, ch, bs_pos, veh_pos, veh_pos,
                                            num_contenders_v2v=num_contenders_v2v,
                                            num_contenders_v2i=num_contenders_v2i,
                                            load_factor=load_factor
                                        )
                                    if success:
                                        net_metrics.record_delay(delay)
                                        pkt_counters.ordinary_recv += 1

                        if total_cl_changes > 0:
                            avg_lifetime = sum(membership_lifetimes) / total_cl_changes
                        else:
                            avg_lifetime = 0.0

                        with open(per_step_csv, 'a') as csv_file:
                            cum_eicd = event_log.lcd_events_up_to(emergency_tracker.amb_positions, time_step)
                            current_ccr = emergency_tracker.cluster_connectivity_ratios[
                                -1] if emergency_tracker.cluster_connectivity_ratios else 0.0
                            csv_file.write(
                                f"{time_step},{ch_changes_this},{cl_changes_this},{total_ch_changes},{total_cl_changes},"
                                f"{avg_lifetime:.2f},{cum_eicd},{sliding_apdr:.4f},{current_ccr:.4f}\n")

                        if not amb_present and ambulance_departed and not ambulance_arrived:
                            emergency_tracker.set_arrival(time_step)
                            ambulance_arrived = True

                            # Stop the simulation immediately after ambulance arrives
                            print(f"Ambulance arrived at t={time_step:.1f}s. Ending simulation.")
                            break

                        ch_of = {}
                        is_linked = {}
                        is_clustered = {}
                        head_of_assign = {}
                        head_valid = {}
                        for vid in vehicle_ids:
                            v = clustering_manager.vehicles.get(vid)
                            if v is None:
                                ch_of[vid] = None
                                is_linked[vid] = False
                                is_clustered[vid] = False
                                head_of_assign[vid] = None
                                head_valid[vid] = False
                                continue
                            ch_id = v.connected_ch_id
                            linked = False
                            if ch_id is not None and ch_id in clustering_manager.vehicles:
                                ch_veh = clustering_manager.vehicles[ch_id]
                                dist = math.hypot(v.position.x - ch_veh.position.x,
                                                  v.position.y - ch_veh.position.y)
                                linked = dist <= COMM_RANGE
                            ch_of[vid] = ch_id if linked else None
                            is_linked[vid] = linked
                            is_clustered[vid] = v.mode in (zbc.VehicleMode.CH, zbc.VehicleMode.CM)
                            head_of_assign[vid] = ch_id
                            head_valid[vid] = (ch_id is not None)

                        link_tracker.update(time_step, vehicle_ids, veh_pos, ch_of, is_linked)
                        assign_tr.update(time_step, vehicle_ids, veh_pos, head_of_assign, head_valid)

                        for vid in vehicle_ids:
                            total_vehicle_steps += 1
                            if is_clustered.get(vid, False):
                                covered_vehicle_sum += 1
                            if is_linked.get(vid, False):
                                link_up_steps += 1

                        prev_zone_ch_dict = curr_zone_ch_dict
                        prev_zone_members_dict = curr_zone_members_dict
                        steps.append(time_step)
                        _step_times.append(_time.perf_counter() - _t0)

        finally:
            try:
                traci.close(wait=False)
            except Exception as e:
                print(f"[WARN] traci.close failed: {e}")
            try:
                import numpy as _np
                if len(_step_times) > 5:
                    _arr = _np.array(_step_times[5:])
                    print(f"[PERF] mean={_arr.mean()*1000:.1f}ms  "
                          f"std={_arr.std()*1000:.1f}ms  "
                          f"max={_arr.max()*1000:.1f}ms  n={len(_arr)}")
            except Exception as _e:
                print(f"[PERF] failed: {_e}")

        link_tracker.finalize(steps[-1] if steps else 0)
        assign_tr.finalize(steps[-1] if steps else 0)
        ch_tenure_tr.finalize(steps[-1] if steps else 0)

        coverage_stability = covered_vehicle_sum / total_vehicle_steps if total_vehicle_steps > 0 else 0.0
        avg_ch_assignment_duration = assign_tr.average()
        link_stability_ratio = link_up_steps / total_vehicle_steps if total_vehicle_steps > 0 else 0.0
        avg_link_duration = link_tracker.average()

        # ---- Aggregate results ----
        avg_ch_lifetime = sum(membership_lifetimes) / len(membership_lifetimes) if membership_lifetimes else 0.0
        avg_cluster_lifetime = avg_ch_lifetime
        avg_ch_changes = total_ch_changes / steps[-1] if steps[-1] > 0 else 0.0
        avg_cl_changes = total_cl_changes / steps[-1] if steps[-1] > 0 else 0.0

        total_clusters_created = sum(birth_counter.values())
        avg_cluster_size = sum(size_list) / len(size_list) if size_list else 0.0

        avg_ch_tenure = ch_tenure_tr.average()
        median_ch_tenure = ch_tenure_tr.median()
        max_ch_tenure = ch_tenure_tr.maximum()

        # Emergency metrics
        emdd_avg, emdd_max, emdd_min = emergency_tracker.get_e2ed_stats()
        ambulance_pdr = pkt_counters.get_emergency_pdr()
        sfr = pkt_counters.get_emergency_sfr()
        emergency_throughput = emergency_tracker.get_emergency_throughput()
        amb_transit_time = emergency_tracker.get_transit_time()
        amb_avg_speed = emergency_tracker.get_avg_speed()
        amb_avg_neighbors = emergency_tracker.get_neighbor_avg()
        eicd_total = event_log.lcd_events(emergency_tracker.amb_positions)

        ccr_values = emergency_tracker.cluster_connectivity_ratios
        if ccr_values:
            ccr_avg = sum(ccr_values) / len(ccr_values)
            ccr_max = max(ccr_values)
            ccr_min = min(ccr_values)
        else:
            ccr_avg = ccr_max = ccr_min = 0.0

        # ---- Energy calculation ----
        per_hop_bytes = (PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES
                         + SECURITY_BYTES + WSMP_HEADER_BYTES)
        total_data_bytes = (pkt_counters.ordinary_hops + pkt_counters.emergency_hops) * per_hop_bytes
        total_control_bytes = net_metrics.ctrl_bytes_sent
        total_tx_bytes = total_data_bytes + total_control_bytes
        AVG_DATA_RATE = 6_000_000
        tx_time = (total_tx_bytes * 8) / AVG_DATA_RATE
        rx_time = tx_time
        idle_time = total_vehicle_seconds - tx_time - rx_time
        if idle_time < 0:
            idle_time = 0.0

        net_metrics.update_energy(tx_time, rx_time, idle_time)
        net_metrics.set_sim_time(steps[-1] if steps else 0)
        summary = net_metrics.get_summary()
        network_pdr = pkt_counters.get_ordinary_pdr()

        n_births = event_log.count('birth')
        n_transfers = event_log.count('transfer')
        n_dissolutions = event_log.count('dissolution')

        # Write aggregate CSV
        with open(agg_csv, 'a') as f:
            f.write(
                f"{args.density},{rep},{current_seed},{avg_ch_lifetime:.2f},{total_ch_changes},{total_cl_changes},"
                f"{avg_ch_changes:.2f},{avg_cl_changes:.2f},{avg_cluster_lifetime:.2f},"
                f"{network_pdr:.4f},{summary['Throughput (bps)']:.0f},{summary['Overhead (Bps)']:.0f},"
                f"{summary['Energy Efficiency (bits/J)']:.0f},{summary['Avg Delay (s)']:.6f},{summary['Jitter (s)']:.6f},"
                f"{total_clusters_created},{avg_cluster_size:.2f},{avg_ch_tenure:.2f},{median_ch_tenure:.2f},{max_ch_tenure:.2f},"
                f"{emdd_avg:.6f},{emdd_max:.6f},{emdd_min:.6f},{eicd_total},{ambulance_pdr:.4f},"
                f"{sfr:.4f},{emergency_throughput:.4f},{amb_transit_time:.2f},{amb_avg_speed:.2f},{amb_avg_neighbors:.2f},{ccr_avg:.4f},"
                f"{ccr_max:.4f},{ccr_min:.4f},{coverage_stability:.4f},{avg_ch_assignment_duration:.2f},{link_stability_ratio:.4f},{avg_link_duration:.2f},"
                f"{emergency_tracker.n_loss},{emergency_tracker.n_late},{n_dissolutions},"
                f"0.00,0.00,0.00,0.00,0.00,"
                f"0.00,0.00,100.00,0,"
                                f"{network_pdr:.4f},{pkt_counters.emergency_sent},{pkt_counters.ordinary_sent},"
                f"{pkt_counters.disconnected_seconds},{pkt_counters.total_ambulance_seconds},"
                f"0.00,0.00,{n_births},{n_transfers}\n"
            )

        print(f"\nRep {rep+1} finished.")
        print(f"EMDD avg: {emdd_avg:.6f}s, EICD total: {eicd_total}, Ambulance PDR: {ambulance_pdr:.4f}")
        print(f"SFR: {sfr:.4f}, Throughput: {emergency_throughput:.4f} pkts/s, Transit: {amb_transit_time:.2f}s, Speed: {amb_avg_speed:.2f} m/s")
        print(f"Avg Neighbors: {amb_avg_neighbors:.2f}, Cluster Connectivity Ratio: {ccr_avg:.4f}")
        for k, v in summary.items():
            print(f"{k}: {v}")

    print(f"\nAll {args.reps} repetitions finished. Results saved to {agg_csv}")

if __name__ == "__main__":
    main()