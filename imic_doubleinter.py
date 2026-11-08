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
from net_metrics import (
    NetworkMetrics,
    MAC_HEADER_BYTES,
    FCS_BYTES,
    SECURITY_BYTES,
    WSMP_HEADER_BYTES,
    PACKET_SIZE_BYTES,
    COMM_RANGE
)
from metrics_tracking import PacketCounters, EpisodeTracker, UnifiedEventLog
import numpy as np

# ----------------------------------------------------------------------
# Constants for overhead accounting (IMICLiVAN uses only HELLO messages)
# ----------------------------------------------------------------------
CTRL_HELLO_BYTES = 64
BEACON_RATE_HZ = 10
CTRL_HELLO_TOTAL_BYTES = CTRL_HELLO_BYTES + MAC_HEADER_BYTES + FCS_BYTES + SECURITY_BYTES + WSMP_HEADER_BYTES

# ----------------------------------------------------------------------
step_length = 1.0

packet_forwarding_betha = 1.2
main_formula_lambda = 0.3
is_neighber_distance_threshold = 100

weighted_distance = 0.1
weighted_velocity = 0.1
weighted_trust = 0.4
weighted_bs_distance_ = 0.1
weighted_answer_ratio = 0.1
weighted_node_degree = 0.1
weighted_node_center = 0.1

num_messages = 10
loss_probability = 0.5
min_delay = 0.3
max_delay = 0.9
_rng_alg_imic = random.Random(0)

def set_imic_algorithm_seed(seed):
    _rng_alg_imic.seed(seed)

# Ambulance constants
AMBULANCE_SPEED = 20.0
AMBULANCE_DEPART_DEFAULT = 20.0
AMBULANCE_ROUTE = "route_Ambulance"
SFR_THRESHOLD_MS = 100.0

# ------------------------------------------------------------
# SUMO setup
# ------------------------------------------------------------
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
    parser = argparse.ArgumentParser(description="IMICLiVAN mixed topology with ambulance.")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--density", type=int, default=100,
                        help="Target average number of vehicles on the road simultaneously (excluding ambulance)")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions")
    parser.add_argument("--amb_depart", type=float, default=AMBULANCE_DEPART_DEFAULT,
                        help="Ambulance departure time in seconds")
    return parser.parse_args()

args = parse_arguments()
TARGET_DENSITY = args.density
RANDOM_SEED = args.seed
AMBULANCE_DEPART = args.amb_depart
BASE_DIR = Path(__file__).resolve().parents[0]

net_file = "network/emergency/double_inter/double_inter.net.xml"
additional_files = "network/emergency/double_inter/additional.xml"

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
        # ---- Ambulance route ----
        ("route_Ambulance", ["arm_W_in", "conn_EB", "arm_E2_out"]),

        # ---- Background routes ----
        ("route_W_to_E",   ["arm_W_in", "conn_EB", "arm_E2_out"]),
        ("route_E_to_W",   ["arm_E2_in", "conn_WB", "arm_W_out"]),
        ("route_W_to_N1",  ["arm_W_in", "arm_N1_out"]),
        ("route_W_to_S1",  ["arm_W_in", "arm_S1_out"]),
        ("route_N1_to_E",  ["arm_N1_in", "conn_EB", "arm_E2_out"]),
        ("route_N1_to_W",  ["arm_N1_in", "arm_W_out"]),          # fixed
        ("route_S1_to_E",  ["arm_S1_in", "conn_EB", "arm_E2_out"]),
        ("route_S1_to_W",  ["arm_S1_in", "arm_W_out"]),          # fixed

        ("route_N2_to_E",  ["arm_N2_in", "arm_E2_out"]),
        ("route_N2_to_W",  ["arm_N2_in", "conn_WB", "arm_W_out"]),
        ("route_S2_to_E",  ["arm_S2_in", "arm_E2_out"]),
        ("route_S2_to_W",  ["arm_S2_in", "conn_WB", "arm_W_out"]),
        ("route_E2_to_N",  ["arm_E2_in", "arm_N2_out"]),
        ("route_E2_to_S",  ["arm_E2_in", "arm_S2_out"]),

        # Cross‑intersection through movements
        ("route_E1_to_N2", ["conn_EB", "arm_N2_out"]),
        ("route_E1_to_S2", ["conn_EB", "arm_S2_out"]),
        ("route_W2_to_N1", ["conn_WB", "arm_N1_out"]),
        ("route_W2_to_S1", ["conn_WB", "arm_S1_out"]),
    ]
    for route_id, edges in routes:
        traci.route.add(route_id, edges)

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

    avg_trip_time = 120.0   # shorter network

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

# ========== IMICLiVAN helper functions ==========
def simulate_delay():
    return _rng_alg_imic.uniform(min_delay, max_delay)

def simulate_packet_loss():
    return _rng_alg_imic.random() < loss_probability

def get_position(x):
    return traci.vehicle.getPosition(x)

def get_distance_2D(a, b):
    return traci.simulation.getDistance2D(
        get_position(a)[0], get_position(a)[1],
        get_position(b)[0], get_position(b)[1], False)

def calculate_distance(vehicle_n, vehicle_ids):
    local_distance = 0
    for v in vehicle_ids:
        dis = math.sqrt((get_position(vehicle_n)[0] - get_position(v)[0]) ** 2 +
                        (get_position(vehicle_n)[1] - get_position(v)[1]) ** 2)
        local_distance += dis
    return local_distance

def check_distance(v1, v2):
    return get_distance_2D(v1, v2) < is_neighber_distance_threshold

def calculate_relative_speed(vehicle_n, vehicle_ids):
    related_speed = 0
    for v in vehicle_ids:
        speed = abs(traci.vehicle.getSpeed(vehicle_n) - traci.vehicle.getSpeed(v))
        related_speed += speed
    return related_speed

def simulate_packeting():
    total_delay = 0
    number_of_dropped_packets = 0
    number_of_forwarded_packets = 0
    for m in range(1, num_messages + 1):
        if simulate_packet_loss():
            continue
        delay = simulate_delay()
        total_delay += delay
        avg_delay = total_delay / m
        if delay <= avg_delay * packet_forwarding_betha:
            number_of_forwarded_packets += 1
        else:
            number_of_dropped_packets += 1
    if num_messages == number_of_dropped_packets:
        return 0
    direct_trust = number_of_forwarded_packets / (num_messages)
    return direct_trust

def calculate_indirect_trust(vehicle_n, vehicle_ids):
    indirect_trust = 0
    for vehicle_m in vehicle_ids:
        if vehicle_n != vehicle_m and check_distance(vehicle_n, vehicle_m):
            direct_trust = simulate_packeting()
            indirect_trust += direct_trust
    return indirect_trust

def calculate_trust(vehicle_n, vehicle_ids):
    direct_trust = simulate_packeting()
    indirect_trust = calculate_indirect_trust(vehicle_n, vehicle_ids)
    trust = main_formula_lambda * direct_trust + (1 - main_formula_lambda) * indirect_trust
    return trust

def calculate_bs_distance(vehicle_n):
    baseStation1_pos = traci.poi.getPosition('baseStation')
    vehicle_position = get_position(vehicle_n)
    distance = math.sqrt(
        (vehicle_position[0] - baseStation1_pos[0]) ** 2 +
        (vehicle_position[1] - baseStation1_pos[1]) ** 2)
    return distance

def calculate_answer_ratio(vehicle_n, vehicle_ids):
    pdr = 0
    packets_sent = 0
    packets_received = 0
    for num_sender_packets in range(5):
        sender_pos = get_position(vehicle_n)
        packets_sent = packets_sent + 1
        for recevier_id in vehicle_ids:
            if recevier_id != vehicle_n:
                recevier_pos = get_position(recevier_id)
                distance = math.sqrt((sender_pos[0] - recevier_pos[0]) ** 2 +
                                     (sender_pos[1] - recevier_pos[1]) ** 2)
                if distance <= COMM_RANGE:
                    packet_loss_prob = distance / COMM_RANGE
                    if _rng_alg_imic.random() >= packet_loss_prob:
                        packets_received = packets_received + 1
                        break
    pdr = (packets_received / packets_sent) if packets_sent > 0 else 0
    return pdr

def calculate_node_degree(vehicle_n, vehicle_ids):
    vehicle_position_n = get_position(vehicle_n)
    vehicles_in_range = []
    for vehicle_m in vehicle_ids:
        if vehicle_m != vehicle_n:
            vehicle_position_m = get_position(vehicle_m)
            distance = traci.simulation.getDistance2D(
                vehicle_position_n[0], vehicle_position_n[1],
                vehicle_position_m[0], vehicle_position_m[1], False)
            if distance <= COMM_RANGE:
                vehicles_in_range.append(vehicle_m)
    return len(vehicles_in_range)

def calculate_node_center(vehicle_n):
    return 0

def vehicle_direction(vehicle_n):
    angle = traci.vehicle.getAngle(vehicle_n) % 360
    if angle < 90:
        return 0
    elif angle < 180:
        return 1
    elif angle < 270:
        return 2
    else:
        return 3

def get_vehicle_data(vehicle_ids):
    dic = {
        'vehicle': [],
        'position_0': [],
        'position_1': [],
        'lane_id': []
    }
    for v in vehicle_ids:
        position = traci.vehicle.getPosition(v)
        dic['vehicle'].append(v)
        dic['position_0'].append(position[0])
        dic['position_1'].append(position[1])
        dic['lane_id'].append(vehicle_direction(v))
    return dic

# ---------- K-means clustering ----------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustering(vehicle_ids):
    vehicle_data = get_vehicle_data(vehicle_ids)
    df = pd.DataFrame(vehicle_data)
    df['lane_id'] = df['lane_id'].astype('float64')
    cdf = df[['position_0', 'position_1', 'lane_id']]
    X = cdf.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(vehicle_ids)
    if n_samples < 2:
        return {0: vehicle_data['vehicle']}
    elif n_samples == 2:
        return {0: vehicle_data['vehicle']}

    max_k = min(11, n_samples - 1)
    if max_k < 2:
        return {0: vehicle_data['vehicle']}

    ClusterNumber = range(2, max_k + 1)
    best_silhoutte_score = -1
    best_k = 2

    for k in ClusterNumber:
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=12, random_state=4)
        k_means.fit(X_scaled)
        labels = k_means.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        if silhouette_avg > best_silhoutte_score:
            best_k = k
            best_silhoutte_score = silhouette_avg

    final_k = best_k
    kmeans = KMeans(init='k-means++', n_clusters=final_k, n_init=12, random_state=4)
    kmeans.fit(X_scaled)
    label = kmeans.labels_

    clusters = {i: [] for i in range(final_k)}
    for v, c in zip(df['vehicle'], label):
        clusters[c].append(v)

    return clusters

def calculate_poss(local_distance, relative_velocity, trust, base_station_distance,
                   answer_ratio, node_degree, node_center):
    return ((weighted_distance * local_distance) +
            (weighted_velocity * relative_velocity) +
            (weighted_trust * trust) +
            (weighted_bs_distance_ * base_station_distance) +
            (weighted_answer_ratio * answer_ratio) +
            (weighted_node_degree * node_degree) +
            (weighted_node_center * node_center))

def cluster_head_selection(vehicle_n, vehicle_ids):
    local_distance = calculate_distance(vehicle_n, vehicle_ids)
    relative_velocity = calculate_relative_speed(vehicle_n, vehicle_ids)
    trust = calculate_trust(vehicle_n, vehicle_ids)
    base_station_distance = calculate_bs_distance(vehicle_n)
    answer_ratio = calculate_answer_ratio(vehicle_n, vehicle_ids)
    node_degree = calculate_node_degree(vehicle_n, vehicle_ids)
    node_center = calculate_node_center(vehicle_n)

    probability_cluster_head = calculate_poss(local_distance, relative_velocity, trust,
                                              base_station_distance, answer_ratio,
                                              node_degree, node_center)
    return probability_cluster_head

# =========================================================
# Persistent cluster matching
# =========================================================
_IMIC_CLUSTER_COUNTER = [0]


def match_clusters(curr_clusters, prev_clusters):
    persistent_curr = {}

    if not prev_clusters:
        for curr_label, curr_members in curr_clusters.items():
            _IMIC_CLUSTER_COUNTER[0] += 1
            pid = f"IMIC_C{_IMIC_CLUSTER_COUNTER[0]}"
            persistent_curr[pid] = curr_members
        return persistent_curr, {}

    consumed_prev_pids = set()
    for curr_label, curr_members in curr_clusters.items():
        member_set = set(curr_members)
        best_jac = 0.0
        best_prev_pid = None
        for prev_pid, prev_members in prev_clusters.items():
            if prev_pid in consumed_prev_pids:
                continue
            prev_set = set(prev_members)
            inter = len(member_set & prev_set)
            union = len(member_set | prev_set)
            if union > 0:
                jac = inter / union
                if jac > best_jac:
                    best_jac = jac
                    best_prev_pid = prev_pid

        if best_prev_pid is not None and best_jac >= 0.5:
            persistent_curr[best_prev_pid] = curr_members
            consumed_prev_pids.add(best_prev_pid)
        else:
            _IMIC_CLUSTER_COUNTER[0] += 1
            pid = f"IMIC_C{_IMIC_CLUSTER_COUNTER[0]}"
            persistent_curr[pid] = curr_members

    return persistent_curr, prev_clusters

# ========== Helper for logging ==========
def get_vehicle_data_log(vehicle_id, rsu_pos):
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

# =========================================================
# Emergency Tracker
# =========================================================
class EmergencyTracker:
    def __init__(self):
        self.amb_positions = {}  # time_step -> (x, y) of ambulance for dynamic EICD

        # EICD – position‑based
        self.change_events = []            # (time, ch_veh_id, ch_pos)

        # Other metrics
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

    def record_ambulance_connectivity(self, ambulance_cluster, curr_members_dict, veh_pos, comm_range):
        if ambulance_cluster is None:
            return
        members = curr_members_dict.get(ambulance_cluster, set())
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
    clustering_dir = os.path.join(base_dir, "metrics", "imiclivan_doubleinter", "clustering_data")
    network_dir = os.path.join(base_dir, "metrics", "imiclivan_doubleinter", "network_data")
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)

    eta_relay = float(os.environ.get('ECDC_ETA_RELAY', '0.15'))
    gamma = float(os.environ.get('ECDC_GAMMA', '0.50'))
    load_factor = float(os.environ.get('ECDC_LOAD_FACTOR', '1.0'))

    amb_depart_int = int(AMBULANCE_DEPART)
    RUN_TAG = os.environ.get('ECDC_RUN_TAG', '').strip()
    TAG_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
    _file_suffix = TAG_SUFFIX
    agg_csv = os.path.join(clustering_dir, f"imiclivan_doubleinter{args.density}_2rsu_depart{amb_depart_int}{_file_suffix}.csv")
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
        _IMIC_CLUSTER_COUNTER[0] = 0
        current_seed = RANDOM_SEED + rep

        random.seed(current_seed)
        np.random.seed(current_seed)
        rng_pkt = random.Random(current_seed * 1000 + 1)
        set_imic_algorithm_seed(current_seed * 1000 + 3)
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

        # Metrics containers
        prev_members_dict = {}
        prev_ch_map = {}
        membership_birth_time = {}
        membership_lifetimes = []
        total_ch_changes = 0
        total_cl_changes = 0
        steps = []

        covered_vehicle_sum = 0.0
        total_vehicle_steps = 0

        link_up_steps = 0
        vehicle_prev_cluster = {}

        total_vehicle_seconds = 0.0

        birth_counter = {}
        size_list = []
        ch_start_time = {}
        ch_end_time = {}

        prev_ambulance_cluster = None
        ambulance_departed = False
        ambulance_arrived = False

        network_log = os.path.join(network_dir,
                                   f"imiclivan_doubleinter{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_2rsu{_file_suffix}.txt")
        per_step_csv = os.path.join(clustering_dir,
                                    f"per_step_imiclivan_doubleinter{args.density}_rep{rep}_seed{current_seed}_depart{amb_depart_int}_2rsu{_file_suffix}.csv")
        with open(per_step_csv, 'w') as csv_file:
            csv_file.write("time_step,ch_changes_this_step,cl_changes_this_step,cumulative_ch_changes,cumulative_cl_changes,avg_lifetime,cumulative_eicd,sliding_apdr,ccr\n")

        emergency_delays_csv = os.path.join(
            clustering_dir,
            f"emergency_delays{args.density}_seed{current_seed}_depart{amb_depart_int}{_file_suffix}.csv")
        with open(emergency_delays_csv, 'w') as edf:
            edf.write("time_step,delay_s,success\n")

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

                    # Log vehicle data
                    for vid in vehicle_ids:
                        data = get_vehicle_data_log(vid, rsu_positions)
                        write_vehicle_data(log_file, time_step, data)
                    log_file.flush()

                    # ---- IMICLiVAN clustering ----
                    if len(vehicle_ids) < 3:
                        curr_members_dict = {}
                        curr_ch_map = {}
                    else:
                        raw_clusters = clustering(vehicle_ids)
                        raw_clusters = {k: v for k, v in raw_clusters.items() if len(v) >= 3}
                        if not raw_clusters:
                            curr_members_dict = {}
                            curr_ch_map = {}
                        else:
                            persistent_curr, _ = match_clusters(raw_clusters, prev_members_dict)
                            curr_members_dict = persistent_curr
                            curr_ch_map = {}
                            for pid, members in curr_members_dict.items():
                                best_ch = None
                                best_score = -1
                                for v in members:
                                    score = cluster_head_selection(v, members)
                                    if score > best_score:
                                        best_score = score
                                        best_ch = v
                                curr_ch_map[pid] = best_ch

                    if amb_present:
                        ambulance_cluster = None
                        for cid, members in curr_members_dict.items():
                            if "ambulance" in members:
                                ambulance_cluster = cid
                                break
                        if ambulance_cluster is None:
                            new_cid = f"IMIC_AMB_{int(time_step)}"
                            curr_members_dict[new_cid] = {"ambulance"}
                            curr_ch_map[new_cid] = "ambulance"
                            ambulance_cluster = new_cid

                    # ----- EICD: record CH changes with positions -----
                    for cid, new_ch in curr_ch_map.items():
                        if new_ch is None:
                            continue
                        old_ch = prev_ch_map.get(cid)
                        if new_ch in vehicle_ids:
                            event_log.note_ch_position(cid,
                                                       traci.vehicle.getPosition(new_ch))
                        if old_ch is None:
                            if new_ch in vehicle_ids:
                                event_log.record(time_step, 'birth', cid,
                                                 traci.vehicle.getPosition(new_ch), ch_id=new_ch)
                        elif old_ch != new_ch:
                            if new_ch in vehicle_ids:
                                event_log.record(time_step, 'transfer', cid,
                                                 traci.vehicle.getPosition(new_ch),
                                                 ch_id=new_ch, old_ch_id=old_ch)

                    for cid, old_ch in prev_ch_map.items():
                        if cid in curr_ch_map or old_ch is None:
                            continue
                        pos = event_log.last_position(cid)
                        if old_ch in vehicle_ids:
                            pos = traci.vehicle.getPosition(old_ch)
                        event_log.record(time_step, 'dissolution', cid, pos,
                                         old_ch_id=old_ch)

                    ch_changes_this = count_ch_changes(prev_ch_map, curr_ch_map)
                    membership_birth_time, membership_lifetimes, cl_changes_this = baseline_style_metrics(
                        prev_members_dict, curr_members_dict,
                        membership_birth_time, membership_lifetimes, time_step
                    )

                    birth_counter = track_cluster_births(curr_members_dict, prev_members_dict,
                                                         birth_counter, time_step)
                    size_list = track_cluster_size(curr_members_dict, size_list)

                    # ---- Overhead recording ----
                    net_metrics.record_control_overhead(
                        len(vehicle_ids) * CTRL_HELLO_TOTAL_BYTES * BEACON_RATE_HZ,
                        num_packets=len(vehicle_ids) * BEACON_RATE_HZ
                    )

                    total_ch_changes += ch_changes_this
                    total_cl_changes += cl_changes_this

                    # ---- Compute neighbor_counts and bs_neighbor_count BEFORE ambulance handling ----
                    veh_pos = {vid: traci.vehicle.getPosition(vid) for vid in vehicle_ids}

                    ch_tenure_tr.update(
                        time_step, vehicle_ids, veh_pos,
                        {ch: cid for cid, ch in curr_ch_map.items() if ch is not None},
                        {ch: True for ch in curr_ch_map.values() if ch is not None},
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

                        # Get ambulance cluster
                        ambulance_cluster = None
                        for cid, members in curr_members_dict.items():
                            if "ambulance" in members:
                                ambulance_cluster = cid
                                break

                        # Record connectivity EVERY step
                        if ambulance_cluster is not None:
                            emergency_tracker.record_ambulance_connectivity(
                                ambulance_cluster, curr_members_dict, veh_pos, COMM_RANGE
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
                            if ambulance_cluster is not None:
                                ch = curr_ch_map.get(ambulance_cluster)

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

                    # ---- Normal application traffic ----
                    if len(vehicle_ids) > 2:
                        veh_to_ch = {}
                        for pid, members in curr_members_dict.items():
                            ch = curr_ch_map.get(pid)
                            if ch:
                                for m in members:
                                    veh_to_ch[m] = ch
                        for ch in curr_ch_map.values():
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

                    # ---- Write per-step CSV ----
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
                        print(f"Ambulance arrived at t={time_step:.1f}s. Ending simulation.")
                        break

                    # ---- Per-vehicle link & assignment tracking ----
                    vehicle_to_cluster = {}
                    for cid, members in curr_members_dict.items():
                        for vid in members:
                            vehicle_to_cluster[vid] = cid

                    ch_of = {}
                    is_linked = {}
                    is_clustered = {}
                    head_of_assign = {}
                    head_valid = {}
                    for vid in vehicle_ids:
                        cluster_id = vehicle_to_cluster.get(vid)
                        ch_id = curr_ch_map.get(cluster_id) if cluster_id is not None else None
                        linked = False
                        if ch_id is not None and ch_id in veh_pos:
                            p_v = veh_pos[vid]
                            p_h = veh_pos[ch_id]
                            linked = math.hypot(p_v[0] - p_h[0], p_v[1] - p_h[1]) <= COMM_RANGE
                        ch_of[vid] = ch_id if linked else None
                        is_linked[vid] = linked
                        is_clustered[vid] = (cluster_id is not None)
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

                    prev_members_dict = curr_members_dict
                    prev_ch_map = curr_ch_map
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