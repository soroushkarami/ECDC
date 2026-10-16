# naive_ea_doubleinter.py
"""
ECDC double-intersection emergency scenario harness.

Two 4-arm intersections connected by a 500 m corridor.
- RSU 1 at (250, 250), RSU 2 at (750, 250)
- Single BS at x = 500
- Ambulance traverses arm_W_in -> conn_EB -> arm_E2_out

Reuses the naive_ea algorithm from naive_ea_clustering.py.
Scenario geometry, routes, and vehicle generation are copied from
lamac_doubleinter.py so comparisons stay apples-to-apples.
"""

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
import naive_ea_clustering as ecdc
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
# Constants for overhead accounting
# ----------------------------------------------------------------------
CTRL_HELLO_BYTES = 64
BEACON_RATE_HZ = 10

CTRL_HELLO_TOTAL_BYTES = CTRL_HELLO_BYTES + MAC_HEADER_BYTES + FCS_BYTES + SECURITY_BYTES + WSMP_HEADER_BYTES

# ----------------------------------------------------------------------
AMBULANCE_SPEED = 20.0
AMBULANCE_DEPART_DEFAULT = 20.0
AMBULANCE_ROUTE = "route_Ambulance"
SFR_THRESHOLD_MS = 100.0

# Optional user-controlled tag to avoid overwriting existing CSVs
RUN_TAG = os.environ.get('ECDC_RUN_TAG', '').strip()
TAG_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""

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
    return os.path.join(sumo_home, "bin", "sumo-gui") if gui else os.path.join(sumo_home, "bin", "sumo")

def parse_arguments():
    parser = argparse.ArgumentParser(description="ECDC double-intersection with ambulance (emergency scenario).")
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

# ---- Double-intersection network ----
net_file = "network/emergency/double_inter/double_inter.net.xml"
additional_files = "network/emergency/double_inter/additional.xml"

sumo_binary = get_sumo_binary(gui=args.gui)
sumo_cmd_base = [
    sumo_binary,
    "--net-file", net_file,
    "--additional-files", additional_files,
    "--step-length", "1.0",
]

# ========== Routes (identical to lamac_doubleinter.py) ==========
def define_routes():
    routes = [
        # Ambulance route (west -> I1 -> connector -> I2 -> hospital)
        ("route_Ambulance", ["arm_W_in", "conn_EB", "arm_E2_out"]),

        # Through movements
        ("route_W_to_E", ["arm_W_in", "conn_EB", "arm_E2_out"]),
        ("route_E_to_W", ["arm_E2_in", "conn_WB", "arm_W_out"]),

        # Intersection 1 turning movements
        ("route_W_to_N1", ["arm_W_in", "arm_N1_out"]),
        ("route_W_to_S1", ["arm_W_in", "arm_S1_out"]),
        ("route_N1_to_E", ["arm_N1_in", "conn_EB", "arm_E2_out"]),
        ("route_N1_to_W", ["arm_N1_in", "arm_W_out"]),
        ("route_S1_to_E", ["arm_S1_in", "conn_EB", "arm_E2_out"]),
        ("route_S1_to_W", ["arm_S1_in", "arm_W_out"]),

        # Intersection 2 turning movements
        ("route_N2_to_E", ["arm_N2_in", "arm_E2_out"]),
        ("route_N2_to_W", ["arm_N2_in", "conn_WB", "arm_W_out"]),
        ("route_S2_to_E", ["arm_S2_in", "arm_E2_out"]),
        ("route_S2_to_W", ["arm_S2_in", "conn_WB", "arm_W_out"]),
        ("route_E2_to_N", ["arm_E2_in", "arm_N2_out"]),
        ("route_E2_to_S", ["arm_E2_in", "arm_S2_out"]),

        # Cross-intersection movements
        ("route_E1_to_N2", ["conn_EB", "arm_N2_out"]),
        ("route_E1_to_S2", ["conn_EB", "arm_S2_out"]),
        ("route_W2_to_N1", ["conn_WB", "arm_N1_out"]),
        ("route_W2_to_S1", ["conn_WB", "arm_S1_out"]),
    ]
    for route_id, edges in routes:
        traci.route.add(route_id, edges)

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

def generate_vehicles(target_density, seed, amb_depart, injection_horizon=400):
    rn = random.Random(seed)
    all_routes = [
        'route_W_to_E', 'route_E_to_W',
        'route_W_to_N1', 'route_W_to_S1',
        'route_N1_to_E', 'route_N1_to_W',
        'route_S1_to_E', 'route_S1_to_W',
        'route_N2_to_E', 'route_N2_to_W',
        'route_S2_to_E', 'route_S2_to_W',
        'route_E2_to_N', 'route_E2_to_S',
        'route_E1_to_N2', 'route_E1_to_S2',
        'route_W2_to_N1', 'route_W2_to_S1',
    ]
    # Shorter trips in the double-intersection topology
    avg_trip_time = 120.0

    total_injection_window = amb_depart + injection_horizon
    total_vehicles = int(math.ceil(target_density * total_injection_window / avg_trip_time))

    print(f"Generating {total_vehicles} vehicles to maintain target density of {target_density} "
          f"(avg trip time = {avg_trip_time:.0f}s, injection window = {total_injection_window:.0f}s)")

    for i in range(total_vehicles):
        veh_id = f'veh{i}'
        speed = rn.uniform(3.611111, 13.888888)
        lane = rn.choice([0, 1])
        route = rn.choice(all_routes)
        depart_time = rn.uniform(0, total_injection_window)
        traci.vehicle.add(veh_id, route, typeID='car',
                          departSpeed=speed, depart=depart_time, departLane=str(lane))

# ========== Helpers ==========
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
        min_dist_to_rsu = min(math.hypot(x - rx, y - ry) for rx, ry in rsu_pos)
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
        dest = ecdc.Position(data['x'] + 1000 * math.cos(rad),
                             data['y'] + 1000 * math.sin(rad))
        is_em = (veh_id == "ambulance")
        if veh_id in manager.vehicles:
            v = manager.vehicles[veh_id]
            v.position = ecdc.Position(data['x'], data['y'])
            v.speed = data['speed']
            v.lane_id = data['lane_id']
            v.edge_id = data['edge_id']
            v.angle = data['angle']
            v.destination = dest
            v.is_emergency = is_em
            v.update_direction()
        else:
            v = ecdc.VehicleState(
                veh_id=veh_id,
                lane_id=data['lane_id'],
                edge_id=data['edge_id'],
                position=ecdc.Position(data['x'], data['y']),
                speed=data['speed'],
                destination=dest,
                is_emergency=is_em,
            )
            v.angle = data['angle']
            v.update_direction()
            manager.add_vehicle(v)

def run_clustering(manager, time_step):
    cluster_ch_dict = manager.run()
    cluster_ch = {cid: info.get("primary_ch") for cid, info in cluster_ch_dict.items()}
    cluster_rch = {cid: info.get("backup_ch") for cid, info in cluster_ch_dict.items()}
    print(f'\n=== Clustering at t={time_step:.1f} s ===')
    for cid, ch_id in cluster_ch.items():
        for rsu in manager.rsus.values():
            if cid in rsu.clusters:
                cl = rsu.clusters[cid]
                member_count = len(cl.core_members) + len(cl.guard_members)
                print(f'    Cluster {cid}: CH={ch_id}, members={member_count}')
                break
    return cluster_ch, cluster_rch

def get_zone_members(manager):
    members = {}
    for rsu in manager.rsus.values():
        for cl in rsu.clusters.values():
            all_members = cl.core_members | cl.guard_members
            cluster_members = {
                vid for vid in all_members
                if vid in manager.vehicles
                and manager.vehicles[vid].mode in (
                    ecdc.VehicleMode.CH,
                    ecdc.VehicleMode.CM,
                    ecdc.VehicleMode.RCH,
                )
            }
            members[cl.cluster_id] = cluster_members
    return members

# ========== Emergency Tracker ==========
class EmergencyTracker:
    def __init__(self):
        self.amb_positions = {}
        self.zone_entries = []
        self.change_events = []
        self.emergency_delays = []
        self.ambulance_packets_sent = 0
        self.ambulance_packets_received = 0
        self.sfr_fail_count = 0
        self.sfr_total = 0
        self.sfr_threshold = SFR_THRESHOLD_MS / 1000.0
        self.n_loss = 0
        self.n_late = 0
        self.n_dissolutions = 0
        self.ambulance_successful_packets = 0
        self.amb_departure_time = None
        self.amb_arrival_time = None
        self.amb_speeds = []
        self.neighbor_counts = []
        self.coverage_fractions = []
        self.cluster_connectivity_ratios = []
        self.eicd_cumulative_over_time = []
        self.ccr_over_time = []
        self.amb_packet_history = []
        self.apdr_over_time = []

        # --- RCH decision statistics ---
        self.rch_decisions = []  # list of dicts, one per emergency-packet step
        self.rch_ever_available = 0  # steps with a valid RCH candidate
        self.rch_none = 0  # steps with no RCH candidate at all
        self.gate_v2v_fail = 0  # RCH existed, failed V2V-hop quality
        self.gate_v2i_fail = 0  # V2V ok, failed V2I improvement
        self.relay_used = 0  # RCH path actually chosen
        self.direct_used = 0  # direct path chosen (with RCH available)
        self.direct_used_no_rch = 0  # direct path chosen (no RCH available)
        self.relay_success = 0
        self.relay_total = 0
        self.direct_success = 0
        self.direct_total = 0
        self.relay_delays = []
        self.direct_delays = []

    def record_rch_decision(self, time_step, rch_available, gate_v2v_pass,
                            gate_v2i_pass, use_rch, success, delay,
                            d_amb_rch, d_amb_bs, d_rch_bs):
        self.rch_decisions.append({
            'time': time_step,
            'rch_available': rch_available,
            'gate_v2v_pass': gate_v2v_pass,
            'gate_v2i_pass': gate_v2i_pass,
            'use_rch': use_rch,
            'success': success,
            'delay': delay,
            'd_amb_rch': d_amb_rch,
            'd_amb_bs': d_amb_bs,
            'd_rch_bs': d_rch_bs,
        })
        if not rch_available:
            self.rch_none += 1
            self.direct_used_no_rch += 1
        else:
            self.rch_ever_available += 1
            if not gate_v2v_pass:
                self.gate_v2v_fail += 1
            if gate_v2v_pass and not gate_v2i_pass:
                self.gate_v2i_fail += 1
            if use_rch:
                self.relay_used += 1
                self.relay_total += 1
                self.relay_delays.append(delay)
                if success:
                    self.relay_success += 1
            else:
                self.direct_used += 1
                self.direct_total += 1
                self.direct_delays.append(delay)
                if success:
                    self.direct_success += 1

    def record_ch_change(self, time_step, ch_vehicle_id, ch_position, kind='transfer'):
        self.change_events.append((time_step, ch_vehicle_id, ch_position))
        if kind == 'dissolution':
            self.n_dissolutions += 1

    def compute_eicd(self):
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
            return (sum(self.emergency_delays) / len(self.emergency_delays),
                    max(self.emergency_delays),
                    min(self.emergency_delays))
        return 0.0, 0.0, 0.0

    def get_neighbor_avg(self):
        if self.neighbor_counts:
            return sum(self.neighbor_counts) / len(self.neighbor_counts)
        return 0.0

    def record_ambulance_connectivity_from_set(self, member_set, veh_pos, comm_range):
        """CCR computed against a raw member set (no mode filter)."""
        if "ambulance" not in member_set:
            return
        total_members = len(member_set) - 1
        if total_members == 0:
            self.cluster_connectivity_ratios.append(0.0)
            return
        amb_pos = veh_pos["ambulance"]
        connected = 0
        for vid in member_set:
            if vid == "ambulance":
                continue
            dist = math.hypot(amb_pos[0] - veh_pos[vid][0],
                              amb_pos[1] - veh_pos[vid][1])
            if dist <= comm_range:
                connected += 1
        self.cluster_connectivity_ratios.append(connected / total_members)

    def get_avg_connectivity(self):
        if self.cluster_connectivity_ratios:
            return sum(self.cluster_connectivity_ratios) / len(self.cluster_connectivity_ratios)
        return 0.0

# ========== Main ==========
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    import time as _time
    _step_times = []
    clustering_dir = os.path.join(base_dir, "metrics", "naive_ea_doubleinter", "clustering_data")
    network_dir = os.path.join(base_dir, "metrics", "naive_ea_doubleinter", "network_data")
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)

    amb_depart_int = int(AMBULANCE_DEPART)
    _abl_suffix = f"_{ecdc.ABLATION_MODE}" if ecdc.ABLATION_MODE else ""
    _file_suffix = _abl_suffix + TAG_SUFFIX
    agg_csv = os.path.join(clustering_dir,
                           f"naive_ea_doubleinter{args.density}_2rsu_depart{amb_depart_int}{_file_suffix}.csv")
    print(f"[ABLATION] mode = {ecdc.ABLATION_MODE or 'NONE'}")
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
                "relay_mean_delay_ms,direct_mean_delay_ms,rch_none_pct,rch_avail_count\n"
            )

    for rep in range(args.reps):
        current_seed = RANDOM_SEED + rep

        random.seed(current_seed)
        np.random.seed(current_seed)
        sumo_cmd = sumo_cmd_base + ["--seed", str(current_seed)]
        if args.gui:
            sumo_cmd.extend(["--delay", "200", "--start"])

        emergency_delays_csv = os.path.join(clustering_dir,
                                            f"emergency_delays{args.density}_seed{current_seed}_depart{amb_depart_int}{_file_suffix}.csv")
        with open(emergency_delays_csv, 'w') as edf:
            edf.write("time_step,delay_s,success\n")

        traci.start(sumo_cmd)

        traci.vehicletype.copy("DEFAULT_VEHTYPE", "car")
        define_routes()
        generate_vehicles(args.density, current_seed, AMBULANCE_DEPART, injection_horizon=400)

        # ---- Ambulance ----
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
            bs_pos = (500.0, 500.0)   # fallback: BS between the two intersections

        ecdc.set_bs_position(ecdc.Position(bs_pos[0], bs_pos[1]))

        net_metrics = NetworkMetrics(sim_time_total=0.0)
        emergency_tracker = EmergencyTracker()

        prev_zone_ch_dict = {}
        prev_zone_members_dict = {}
        membership_birth_time = {}
        membership_lifetimes = []
        total_ch_changes = 0
        total_cl_changes = 0
        steps = []

        covered_vehicle_sum = 0.0
        total_vehicle_steps = 0
        ch_assignment_start = {}
        ch_assignment_durations = []
        link_start_time = {}
        link_durations = []
        link_up_steps = 0
        vehicle_prev_zone = {}

        total_vehicle_seconds = 0.0

        birth_counter = {}
        size_list = []
        ch_start_time = {}
        ch_end_time = {}

        prev_ambulance_zone = None
        ambulance_departed = False
        ambulance_arrived = False

        clustering_manager = ecdc.ClusteringManager(
            dsrc_v2v_range=COMM_RANGE,
            delta_t=1.0,
            vehicle_length_l=5.0,
            A=0.5,
            B=0.5,
        )

        # ---- Two RSUs, one per intersection centre ----
        intersection_centers = [(250.0, 250.0), (750.0, 250.0)]
        for idx, (x, y) in enumerate(intersection_centers):
            rsu = ecdc.RSU(
                rsu_id=f'rsu_{idx}',
                position=ecdc.Position(x, y),
                coverage_range=COMM_RANGE,
                hysteresis_threshold=0.5,
                min_dwell_time=6.0,
                ert_handover_threshold=3.0,
                guard_ratio=0.05,
                overlap_ratio=0.15
            )
            clustering_manager.add_rsu(rsu)
            # ECDC does not use fixed zones, so no configure_* calls are needed.

        network_log = os.path.join(
            network_dir,
            f"naive_ea_doubleinter{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_2rsu{TAG_SUFFIX}.txt"
        )
        per_step_csv = os.path.join(
            clustering_dir,
            f"per_step_naive_ea_doubleinter{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_2rsu{_file_suffix}.csv"
        )
        with open(per_step_csv, 'w') as csv_file:
            csv_file.write("time_step,ch_changes_this_step,cl_changes_this_step,cumulative_ch_changes,"
                           "cumulative_cl_changes,avg_lifetime,cumulative_eicd,sliding_apdr,ccr\n")

        try:
            with open(network_log, 'w', encoding='utf-8') as log_file:
                write_header(log_file)

                step = 0
                while traci.simulation.getMinExpectedNumber() > 0:
                    _t0 = _time.perf_counter()
                    traci.simulationStep()
                    step += 1
                    time_step = traci.simulation.getTime()
                    vehicle_ids = traci.vehicle.getIDList()
                    total_vehicle_seconds += len(vehicle_ids) * 1.0

                    load_factor = 1.0
                    sliding_apdr = 0.0

                    amb_present = "ambulance" in vehicle_ids
                    if amb_present and not ambulance_departed:
                        emergency_tracker.set_departure(time_step)
                        ambulance_departed = True

                    for vid in vehicle_ids:
                        data = get_vehicle_data(vid, rsu_positions)
                        write_vehicle_data(log_file, time_step, data)
                    log_file.flush()

                    update_clustering_manager_state(clustering_manager, vehicle_ids, rsu_positions)

                    curr_zone_ch_dict, curr_zone_rch_dict = run_clustering(clustering_manager, time_step)
                    curr_zone_members_dict = get_zone_members(clustering_manager)

                    # Add ambulance to cluster dictionaries
                    if amb_present:
                        amb_veh = clustering_manager.vehicles.get("ambulance")
                        ambulance_zone = amb_veh.cluster_id if amb_veh else None

                        if ambulance_zone is None:
                            amb_pos = traci.vehicle.getPosition("ambulance")
                            emergency_tracker.record_ambulance_pos(time_step, amb_pos)
                            for rsu in clustering_manager.rsus.values():
                                for cl in rsu.clusters.values():
                                    if cl.cluster_type == "emergency":
                                        ambulance_zone = cl.cluster_id
                                        break
                                if ambulance_zone:
                                    break

                        if ambulance_zone is not None:
                            owner_rsu = None
                            for rsu in clustering_manager.rsus.values():
                                if ambulance_zone in rsu.clusters:
                                    owner_rsu = rsu
                                    break

                            if amb_veh:
                                amb_veh.cluster_id = ambulance_zone
                                amb_veh.rsu_id = owner_rsu.rsu_id if owner_rsu else None

                            if ambulance_zone not in curr_zone_members_dict:
                                curr_zone_members_dict[ambulance_zone] = {"ambulance"}
                            else:
                                curr_zone_members_dict[ambulance_zone].add("ambulance")

                            if ambulance_zone not in curr_zone_ch_dict or curr_zone_ch_dict[ambulance_zone] is None:
                                curr_zone_ch_dict[ambulance_zone] = "ambulance"
                                if amb_veh:
                                    amb_veh.mode = ecdc.VehicleMode.CH
                                    amb_veh.connected_ch_id = "ambulance"

                    # ---- EICD: record CH changes with positions ----
                    for zone_id, new_ch in curr_zone_ch_dict.items():
                        old_ch = prev_zone_ch_dict.get(zone_id)
                        if old_ch != new_ch and new_ch is not None:
                            if new_ch in vehicle_ids:
                                pos = traci.vehicle.getPosition(new_ch)
                                emergency_tracker.record_ch_change(time_step, new_ch, pos, kind='transfer')
                    for zone_id, old_ch in prev_zone_ch_dict.items():
                        if zone_id not in curr_zone_ch_dict and old_ch is not None:
                            if old_ch in vehicle_ids:
                                pos = traci.vehicle.getPosition(old_ch)
                                emergency_tracker.record_ch_change(time_step, old_ch, pos, kind='dissolution')

                    ch_changes_this = count_ch_changes(prev_zone_ch_dict, curr_zone_ch_dict)
                    membership_birth_time, membership_lifetimes, cl_changes_this = baseline_style_metrics(
                        prev_zone_members_dict, curr_zone_members_dict,
                        membership_birth_time, membership_lifetimes, time_step
                    )
                    birth_counter = track_cluster_births(curr_zone_members_dict, prev_zone_members_dict,
                                                         birth_counter, time_step)
                    size_list = track_cluster_size(curr_zone_members_dict, size_list)
                    ch_start_time, ch_end_time = track_ch_tenure(curr_zone_ch_dict, prev_zone_ch_dict,
                                                                 ch_start_time, ch_end_time, time_step)

                    # ---- Overhead (ECDC) ----
                    net_metrics.record_control_overhead(
                        len(vehicle_ids) * CTRL_HELLO_TOTAL_BYTES * BEACON_RATE_HZ,
                        num_packets=len(vehicle_ids) * BEACON_RATE_HZ
                    )

                    total_ch_changes += ch_changes_this
                    total_cl_changes += cl_changes_this

                    veh_pos = {vid: traci.vehicle.getPosition(vid) for vid in vehicle_ids}

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
                            if math.hypot(pos_v[0] - pos_o[0], pos_v[1] - pos_o[1]) <= COMM_RANGE:
                                count += 1
                        neighbor_counts[vid] = count

                    bs_neighbor_count = 0
                    bs_pos_xy = bs_pos
                    for vid in vehicle_ids:
                        if vid == "ambulance":
                            continue
                        pos_v = veh_pos[vid]
                        if math.hypot(pos_v[0] - bs_pos_xy[0], pos_v[1] - bs_pos_xy[1]) <= COMM_RANGE:
                            bs_neighbor_count += 1

                    # ---- Ambulance handling ----
                    if amb_present:
                        amb_pos = traci.vehicle.getPosition("ambulance")
                        amb_speed = traci.vehicle.getSpeed("ambulance")
                        emergency_tracker.record_ambulance_speed(amb_speed)
                        emergency_tracker.record_ambulance_pos(time_step, amb_pos)

                        amb_veh = clustering_manager.vehicles.get("ambulance")
                        ambulance_zone = amb_veh.cluster_id if amb_veh else None

                        # CCR: use the FULL EC membership (bypass mode filter)
                        if ambulance_zone is not None:
                            full_ec_members = set()
                            for rsu in clustering_manager.rsus.values():
                                if ambulance_zone in rsu.clusters:
                                    ec = rsu.clusters[ambulance_zone]
                                    full_ec_members = set(ec.core_members | ec.guard_members)
                                    break
                            emergency_tracker.record_ambulance_connectivity_from_set(
                                full_ec_members, veh_pos, COMM_RANGE
                            )

                        if int(time_step) % 1 == 0 and ambulance_zone is not None:
                            ch = curr_zone_ch_dict.get(ambulance_zone)
                            rch = curr_zone_rch_dict.get(ambulance_zone)

                            # --- RCH decision logic with instrumentation ---
                            rch_available = (rch is not None and rch in veh_pos and rch != "ambulance")
                            gate_v2v_pass = False
                            gate_v2i_pass = False
                            use_rch = False
                            d_amb_rch = float('nan')
                            d_amb_bs = float('nan')
                            d_rch_bs = float('nan')

                            if (ecdc.ABLATION_MODE != 'directbs' and rch_available):
                                rch_pos = veh_pos[rch]
                                d_amb_rch = math.hypot(amb_pos[0] - rch_pos[0],
                                                       amb_pos[1] - rch_pos[1])
                                d_amb_bs = math.hypot(amb_pos[0] - bs_pos[0],
                                                      amb_pos[1] - bs_pos[1])
                                d_rch_bs = math.hypot(rch_pos[0] - bs_pos[0],
                                                      rch_pos[1] - bs_pos[1])

                                if ecdc.ABLATION_MODE == 'nogate':
                                    gate_v2v_pass = True
                                    gate_v2i_pass = True
                                    use_rch = True
                                else:
                                    gate_v2v_pass = d_amb_rch < 0.5 * COMM_RANGE
                                    gate_v2i_pass = d_rch_bs < (1.0 - 0.15) * d_amb_bs
                                    if gate_v2v_pass and gate_v2i_pass:
                                        use_rch = True

                            # Count neighbors of ambulance (for coverage metric)
                            neighbors = 0
                            total_excl = len(vehicle_ids) - 1
                            amb_x, amb_y = amb_pos
                            for vid in vehicle_ids:
                                if vid == 'ambulance':
                                    continue
                                vx, vy = veh_pos[vid]
                                if math.hypot(vx - amb_x, vy - amb_y) < COMM_RANGE:
                                    neighbors += 1

                            if use_rch:
                                num_contenders_v2v = neighbor_counts.get(rch, 0)
                                num_contenders_v2i = bs_neighbor_count
                                success, delay, _ = net_metrics.simulate_application_report(
                                    "ambulance", rch, bs_pos, veh_pos, veh_pos,
                                    num_contenders_v2v=num_contenders_v2v,
                                    num_contenders_v2i=num_contenders_v2i,
                                    load_factor=load_factor,
                                )
                            else:
                                num_contenders_v2i = bs_neighbor_count
                                success, delay, _ = net_metrics.simulate_application_report(
                                    "ambulance", "ambulance", bs_pos, veh_pos, veh_pos,
                                    num_contenders=num_contenders_v2i,
                                    load_factor=load_factor,
                                )

                            emergency_tracker.record_emergency_packet(delay, success, neighbors, total_excl)
                            emergency_tracker.record_rch_decision(
                                time_step, rch_available, gate_v2v_pass, gate_v2i_pass,
                                use_rch, success, delay, d_amb_rch, d_amb_bs, d_rch_bs
                            )
                            # Add per-packet delay logging:
                            with open(os.path.join(clustering_dir,
                                                   f"emergency_delays{args.density}_seed{current_seed}_depart{amb_depart_int}{_file_suffix}.csv"),
                                      'a') as edf:
                                edf.write(f"{time_step},{delay},{1 if success else 0}\n")
                            emergency_tracker.amb_packet_history.append((time_step, success))
                            cutoff_time = time_step - 5
                            emergency_tracker.amb_packet_history = [
                                (t, s) for t, s in emergency_tracker.amb_packet_history if t >= cutoff_time
                            ]
                            if emergency_tracker.amb_packet_history:
                                sliding_success = sum(1 for _, s in emergency_tracker.amb_packet_history if s)
                                sliding_total = len(emergency_tracker.amb_packet_history)
                                sliding_apdr = sliding_success / sliding_total
                            else:
                                sliding_apdr = 0

                    # ---- Normal application traffic ----
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
                            src = random.choice(vehicle_ids)
                            if src == "ambulance":
                                continue
                            ch = veh_to_ch.get(src)
                            net_metrics.packets_sent += 1
                            if ch:
                                hops = 1 if src == ch else 2
                                net_metrics.record_data_hops(hops)
                                if src == ch:
                                    success, delay, _ = net_metrics.simulate_application_report(
                                        src, ch, bs_pos, veh_pos, veh_pos,
                                        num_contenders=bs_neighbor_count,
                                        load_factor=load_factor
                                    )
                                else:
                                    success, delay, _ = net_metrics.simulate_application_report(
                                        src, ch, bs_pos, veh_pos, veh_pos,
                                        num_contenders_v2v=neighbor_counts.get(ch, 0),
                                        num_contenders_v2i=bs_neighbor_count,
                                        load_factor=load_factor
                                    )
                                if success:
                                    net_metrics.record_delay(delay)

                    # ---- Per-step CSV ----
                    if total_cl_changes > 0:
                        avg_lifetime = sum(membership_lifetimes) / total_cl_changes
                    else:
                        avg_lifetime = 0.0
                    with open(per_step_csv, 'a') as csv_file:
                        cum_eicd = emergency_tracker.get_cumulative_eicd_up_to(time_step)
                        current_ccr = (emergency_tracker.cluster_connectivity_ratios[-1]
                                       if emergency_tracker.cluster_connectivity_ratios else 0.0)
                        csv_file.write(
                            f"{time_step},{ch_changes_this},{cl_changes_this},{total_ch_changes},{total_cl_changes},"
                            f"{avg_lifetime:.2f},{cum_eicd},{sliding_apdr:.4f},{current_ccr:.4f}\n")

                    if not amb_present and ambulance_departed and not ambulance_arrived:
                        emergency_tracker.set_arrival(time_step)
                        ambulance_arrived = True
                        print(f"Ambulance arrived at t={time_step:.1f}s. Ending simulation.")
                        break

                    # ---- Per-vehicle metrics ----
                    for vid in vehicle_ids:
                        v = clustering_manager.vehicles.get(vid)
                        if v is None:
                            continue
                        total_vehicle_steps += 1

                        if v.mode in (ecdc.VehicleMode.CH,
                                      ecdc.VehicleMode.CM,
                                      ecdc.VehicleMode.RCH):
                            covered_vehicle_sum += 1

                        ch_id = v.connected_ch_id
                        if ch_id is not None and ch_id in clustering_manager.vehicles:
                            ch_veh = clustering_manager.vehicles[ch_id]
                            dist = ecdc.euclidean_distance_pos(v.position, ch_veh.position)
                            link_up = (dist <= COMM_RANGE)
                        else:
                            link_up = False

                        if link_up:
                            link_up_steps += 1
                            if vid not in link_start_time:
                                link_start_time[vid] = time_step
                        else:
                            if vid in link_start_time:
                                duration = time_step - link_start_time.pop(vid)
                                if duration > 0:
                                    link_durations.append(duration)

                        current_assignment = (vid, ch_id) if ch_id is not None else None
                        prev_assignment = getattr(v, '_prev_assignment', None)
                        if current_assignment != prev_assignment:
                            if prev_assignment is not None and prev_assignment in ch_assignment_start:
                                start = ch_assignment_start.pop(prev_assignment)
                                duration = time_step - start
                                if duration > 0:
                                    ch_assignment_durations.append(duration)
                            if current_assignment is not None:
                                ch_assignment_start[current_assignment] = time_step
                            v._prev_assignment = current_assignment

                    prev_zone_ch_dict = curr_zone_ch_dict
                    prev_zone_members_dict = curr_zone_members_dict
                    steps.append(time_step)
                    _step_times.append(_time.perf_counter() - _t0)

        finally:
            traci.close()
            try:
                import numpy as _np
                if len(_step_times) > 5:
                    _arr = _np.array(_step_times[5:])
                    print(f"[PERF] mean={_arr.mean()*1000:.1f}ms  "
                          f"std={_arr.std()*1000:.1f}ms  "
                          f"max={_arr.max()*1000:.1f}ms  n={len(_arr)}")
            except Exception as _e:
                print(f"[PERF] failed: {_e}")

        # ---- Close remaining assignments/links ----
        for (vid, ch_id), start in ch_assignment_start.items():
            duration = steps[-1] - start
            if duration > 0:
                ch_assignment_durations.append(duration)
        for vid, start in link_start_time.items():
            duration = steps[-1] - start
            if duration > 0:
                link_durations.append(duration)

        coverage_stability = covered_vehicle_sum / total_vehicle_steps if total_vehicle_steps > 0 else 0.0
        avg_ch_assignment_duration = (sum(ch_assignment_durations) / len(ch_assignment_durations)
                                      if ch_assignment_durations else 0.0)
        link_stability_ratio = link_up_steps / total_vehicle_steps if total_vehicle_steps > 0 else 0.0
        avg_link_duration = sum(link_durations) / len(link_durations) if link_durations else 0.0

        avg_ch_lifetime = sum(membership_lifetimes) / len(membership_lifetimes) if membership_lifetimes else 0.0
        avg_cluster_lifetime = avg_ch_lifetime
        avg_ch_changes = total_ch_changes / steps[-1] if steps[-1] > 0 else 0.0
        avg_cl_changes = total_cl_changes / steps[-1] if steps[-1] > 0 else 0.0

        total_clusters_created = sum(birth_counter.values())
        avg_cluster_size = sum(size_list) / len(size_list) if size_list else 0.0

        tenures = compute_ch_tenure_histogram(ch_start_time, ch_end_time, steps[-1] if steps else 0)
        if steps:
            for ch_veh in set(curr_zone_ch_dict.values()) - set(ch_end_time.keys()):
                if ch_veh in ch_start_time:
                    tenures.append(steps[-1] - ch_start_time[ch_veh])
        if tenures:
            avg_ch_tenure = sum(tenures) / len(tenures)
            median_ch_tenure = sorted(tenures)[len(tenures) // 2]
            max_ch_tenure = max(tenures)
        else:
            avg_ch_tenure = median_ch_tenure = max_ch_tenure = 0.0

        emdd_avg, emdd_max, emdd_min = emergency_tracker.get_e2ed_stats()
        ambulance_pdr = emergency_tracker.get_stp()
        sfr = emergency_tracker.get_sfr()
        emergency_throughput = emergency_tracker.get_emergency_throughput()
        amb_transit_time = emergency_tracker.get_transit_time()
        amb_avg_speed = emergency_tracker.get_avg_speed()
        amb_avg_neighbors = emergency_tracker.get_neighbor_avg()
        eicd_total = emergency_tracker.compute_eicd()

        ccr_values = emergency_tracker.cluster_connectivity_ratios
        if ccr_values:
            ccr_avg = sum(ccr_values) / len(ccr_values)
            ccr_max = max(ccr_values)
            ccr_min = min(ccr_values)
        else:
            ccr_avg = ccr_max = ccr_min = 0.0

        total_data_bytes = net_metrics.total_data_hops * (PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES + SECURITY_BYTES + WSMP_HEADER_BYTES)
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

        # --- RCH decision statistics (persisted to CSV) ---
        _avail = max(emergency_tracker.rch_ever_available, 1)
        _total_decisions = (emergency_tracker.rch_ever_available
                            + emergency_tracker.rch_none)
        _total_decisions = max(_total_decisions, 1)
        _relay_tot = max(emergency_tracker.relay_total, 1)
        _direct_tot = max(emergency_tracker.direct_total, 1)
        rch_util_pct = 100.0 * emergency_tracker.relay_used / _avail
        gate_v2v_fail_pct = 100.0 * emergency_tracker.gate_v2v_fail / _avail
        gate_v2i_fail_pct = 100.0 * emergency_tracker.gate_v2i_fail / _avail
        relay_succ_pct = 100.0 * emergency_tracker.relay_success / _relay_tot
        direct_succ_pct = 100.0 * emergency_tracker.direct_success / _direct_tot
        relay_mean_delay_ms = 1000.0 * (sum(emergency_tracker.relay_delays)
                                        / max(len(emergency_tracker.relay_delays), 1))
        direct_mean_delay_ms = 1000.0 * (sum(emergency_tracker.direct_delays)
                                         / max(len(emergency_tracker.direct_delays), 1))
        rch_none_pct = 100.0 * emergency_tracker.rch_none / _total_decisions
        rch_avail_count = emergency_tracker.rch_ever_available

        with open(agg_csv, 'a') as f:
            f.write(
                f"{args.density},{rep},{current_seed},{avg_ch_lifetime:.2f},{total_ch_changes},{total_cl_changes},"
                f"{avg_ch_changes:.2f},{avg_cl_changes:.2f},{avg_cluster_lifetime:.2f},"
                f"{summary['PDR']:.4f},{summary['Throughput (bps)']:.0f},{summary['Overhead (Bps)']:.0f},"
                f"{summary['Energy Efficiency (bits/J)']:.0f},{summary['Avg Delay (s)']:.6f},{summary['Jitter (s)']:.6f},"
                f"{total_clusters_created},{avg_cluster_size:.2f},{avg_ch_tenure:.2f},{median_ch_tenure:.2f},{max_ch_tenure:.2f},"
                f"{emdd_avg:.6f},{emdd_max:.6f},{emdd_min:.6f},{eicd_total},{ambulance_pdr:.4f},"
                f"{sfr:.4f},{emergency_throughput:.4f},{amb_transit_time:.2f},{amb_avg_speed:.2f},{amb_avg_neighbors:.2f},{ccr_avg:.4f},"
                f"{ccr_max:.4f},{ccr_min:.4f},{coverage_stability:.4f},{avg_ch_assignment_duration:.2f},{link_stability_ratio:.4f},{avg_link_duration:.2f},"
                f"{emergency_tracker.n_loss},{emergency_tracker.n_late},{emergency_tracker.n_dissolutions},"
                f"{rch_util_pct:.2f},{gate_v2v_fail_pct:.2f},{gate_v2i_fail_pct:.2f},"
                f"{relay_succ_pct:.2f},{direct_succ_pct:.2f},"
                f"{relay_mean_delay_ms:.2f},{direct_mean_delay_ms:.2f},"
                f"{rch_none_pct:.2f},{rch_avail_count}\n"
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