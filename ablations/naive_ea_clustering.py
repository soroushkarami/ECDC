"""
naive_ea_clustering.py

External emergency-oriented comparator for ECDC.

Design (vs. ECDC):
  - Fixed emergency-cluster radius (R0 = 100 m, no speed adaptation)
  - RCH elected by a single nearest-to-BS rule (no five-factor score)
  - No cross-cluster optimization (no hysteresis)
  - Same forwarding policy (V2V-hop gate + V2I improvement gate)
  - Same EC membership filter, same regular clustering, same Jaccard inheritance

This isolates ECDC's specific machinery (adaptive radius, five-factor RCH,
cross-cluster reassignment) against a simpler emergency-anchored design.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# =========================================================
# naive_ea tuning parameters
# =========================================================
EMERGENCY_CLUSTER_RADIUS    = 100.0   # m (fixed, no adaptation)
REGULAR_CLUSTER_RADIUS      = 150.0   # m
REGULAR_CLUSTER_MIN_MEMBERS = 3

# Harness compatibility (some scripts check this attribute)
ABLATION_MODE = ''


# =========================================================
# Helpers
# =========================================================
def euclidean_distance_pos(a: "Position", b: "Position") -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def calculate_direction_angle_and_type(cx, cy, dx, dy):
    ang = math.degrees(math.atan2(dy - cy, dx - cx))
    while ang < 0:
        ang += 360.0
    while ang >= 360.0:
        ang -= 360.0
    if 0.0 <= ang < 90.0:
        return ang, 1
    if 90.0 <= ang < 180.0:
        return ang, 2
    if 180.0 <= ang < 270.0:
        return ang, 3
    return ang, 4


# =========================================================
# BS position (module-level, set by harness)
# =========================================================
BS_POSITION: Optional["Position"] = None


def set_bs_position(pos: "Position") -> None:
    global BS_POSITION
    BS_POSITION = pos


# =========================================================
# Enums and messages
# =========================================================
class VehicleMode(Enum):
    UM  = "UM"
    CH  = "CH"
    CM  = "CM"
    RCH = "RCH"


@dataclass
class Position:
    x: float
    y: float


@dataclass
class HelloMessage:
    veh_id: str
    lane_id: str
    edge_id: str
    x_n: float
    y_n: float
    v_n: float
    dir_n: int
    is_emergency: bool = False


# =========================================================
# VehicleState (same shape as ECDC)
# =========================================================
@dataclass(init=False)
class VehicleState:
    veh_id: str
    lane_id: str
    edge_id: str
    position: Position
    speed: float
    destination: Position
    theta_n: float = 0.0
    dir_n: int = 1
    mode: VehicleMode = VehicleMode.UM

    rsu_id: Optional[str] = None
    cluster_id: Optional[str] = None
    connected_ch_id: Optional[str] = None
    one_hop_neighbors: Set[str] = field(default_factory=set)

    acceleration: float = 0.0
    angle: float = 0.0
    nearest_rsu_id: Optional[str] = None
    neighbor_count: int = 0
    is_emergency: bool = False

    previous_position: Position = field(default_factory=lambda: Position(0.0, 0.0))
    previous_speed: float = 0.0
    previous_cluster_id: Optional[str] = None
    previous_ch_id: Optional[str] = None
    cluster_entry_time: float = 0.0
    membership_start_time: float = 0.0
    last_membership_change_time: float = 0.0
    ch_role_start_time: float = 0.0
    last_ch_change_time: float = 0.0
    in_guard_zone: bool = False
    handover_candidate_cluster: Optional[str] = None
    handover_condition_start_time: Optional[float] = None
    membership_change_count: int = 0
    ch_change_count: int = 0
    accumulated_membership_stable_time: float = 0.0
    accumulated_ch_stable_time: float = 0.0

    sh_avg: float = 0.0
    v_n_value: float = 0.0
    is_dead_member: bool = False
    predicted_ert: float = 0.0

    def __init__(self, veh_id=None, lane_id="", edge_id="", position=None, speed=0.0,
                 destination=None, theta_n=0.0, dir_n=1, mode=VehicleMode.UM,
                 rsu_id=None, cluster_id=None, connected_ch_id=None,
                 is_emergency=False):
        if veh_id is None:
            raise ValueError("VehicleState requires veh_id.")
        self.veh_id = veh_id
        self.lane_id = lane_id
        self.edge_id = edge_id
        self.position = position if position is not None else Position(0.0, 0.0)
        self.speed = speed
        self.destination = destination if destination is not None else Position(0.0, 0.0)
        self.theta_n = theta_n
        self.dir_n = dir_n
        self.mode = mode
        self.rsu_id = rsu_id
        self.cluster_id = cluster_id
        self.connected_ch_id = connected_ch_id
        self.is_emergency = is_emergency
        self.one_hop_neighbors = set()
        self.acceleration = 0.0
        self.angle = 0.0
        self.nearest_rsu_id = None
        self.neighbor_count = 0
        self.previous_position = Position(self.position.x, self.position.y)
        self.previous_speed = self.speed
        self.previous_cluster_id = None
        self.previous_ch_id = None
        self.cluster_entry_time = 0.0
        self.membership_start_time = 0.0
        self.last_membership_change_time = 0.0
        self.ch_role_start_time = 0.0
        self.last_ch_change_time = 0.0
        self.in_guard_zone = False
        self.handover_candidate_cluster = None
        self.handover_condition_start_time = None
        self.membership_change_count = 0
        self.ch_change_count = 0
        self.accumulated_membership_stable_time = 0.0
        self.accumulated_ch_stable_time = 0.0
        self.sh_avg = 0.0
        self.v_n_value = 0.0
        self.is_dead_member = False
        self.predicted_ert = 0.0
        self.update_direction()

    def update_direction(self):
        self.theta_n, self.dir_n = calculate_direction_angle_and_type(
            self.position.x, self.position.y,
            self.destination.x, self.destination.y)

    def update_motion_state(self, delta_t):
        if delta_t > 0:
            self.acceleration = (self.speed - self.previous_speed) / delta_t
        else:
            self.acceleration = 0.0
        dx = self.position.x - self.previous_position.x
        dy = self.position.y - self.previous_position.y
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            self.angle = math.degrees(math.atan2(dy, dx))
        self.previous_position = Position(self.position.x, self.position.y)
        self.previous_speed = self.speed
        self.update_direction()

    def build_hello_msg(self):
        return HelloMessage(
            veh_id=self.veh_id, lane_id=self.lane_id, edge_id=self.edge_id,
            x_n=self.position.x, y_n=self.position.y, v_n=self.speed,
            dir_n=self.dir_n, is_emergency=self.is_emergency)


# =========================================================
# Cluster
# =========================================================
@dataclass
class Cluster:
    cluster_id: str
    cluster_type: str
    center: Position
    radius: float
    ch_id: Optional[str] = None
    relay_ch_id: Optional[str] = None
    core_members: Set[str] = field(default_factory=set)
    guard_members: Set[str] = field(default_factory=set)
    structural_active_time: float = 0.0
    creation_time: float = 0.0


# =========================================================
# RSU
# =========================================================
class RSU:
    def __init__(self, rsu_id, position, coverage_range=900.0,
                 coverage_radius=None, hello_interval=1.0, **kwargs):
        self.rsu_id = rsu_id
        self.position = position
        self.coverage_range = coverage_radius if coverage_radius is not None else coverage_range
        self.coverage_radius = self.coverage_range
        self.hello_interval = hello_interval
        self.clusters: Dict[str, Cluster] = {}
        self.received_hello_msgs: Dict[str, HelloMessage] = {}
        self._regular_cluster_counter = 0
        self._prev_regular_clusters: List[Cluster] = []
        self.lane_geometries: Dict[str, Tuple[Position, Position]] = {}

    # ---------- Legacy no-ops ----------
    def set_lane_geometry(self, lane_id, start_pos, end_pos):
        self.lane_geometries[lane_id] = (start_pos, end_pos)

    def configure_lane_zones(self, *args, **kwargs):
        pass

    def configure_zones(self, *args, **kwargs):
        pass

    def in_coverage(self, vehicle):
        return euclidean_distance_pos(self.position, vehicle.position) <= self.coverage_range

    def receive_hello_msg(self, msg):
        self.received_hello_msgs[msg.veh_id] = msg

    # ---------- Emergency cluster (fixed radius) ----------
    def update_emergency_cluster(self, ambulance, vehicles, current_time):
        if not self.in_coverage(ambulance):
            for cid in list(self.clusters.keys()):
                if self.clusters[cid].cluster_type == "emergency":
                    del self.clusters[cid]
            return None

        ec_id = f"{self.rsu_id}:EC"
        if ec_id not in self.clusters:
            self.clusters[ec_id] = Cluster(
                cluster_id=ec_id, cluster_type="emergency",
                center=Position(ambulance.position.x, ambulance.position.y),
                radius=EMERGENCY_CLUSTER_RADIUS,
                ch_id=ambulance.veh_id, creation_time=current_time)

        ec = self.clusters[ec_id]
        ec.center = Position(ambulance.position.x, ambulance.position.y)
        ec.radius = EMERGENCY_CLUSTER_RADIUS  # fixed
        ec.ch_id = ambulance.veh_id
        ec.core_members.clear()
        ec.guard_members.clear()
        ec.core_members.add(ambulance.veh_id)
        ambulance.cluster_id = ec_id
        ambulance.rsu_id = self.rsu_id

        for vid, v in vehicles.items():
            if vid == ambulance.veh_id or v.is_emergency:
                continue
            if not self.in_coverage(v):
                continue
            if (euclidean_distance_pos(v.position, ambulance.position)
                    <= EMERGENCY_CLUSTER_RADIUS
                    and v.dir_n == ambulance.dir_n):
                ec.core_members.add(vid)
                v.cluster_id = ec_id
                v.rsu_id = self.rsu_id
        return ec

    # ---------- Regular clusters (same as ECDC, incl. Jaccard) ----------
    def update_regular_clusters(self, vehicles, current_time,
                                exclude_vids=None, allowed_vids=None):
        if exclude_vids is None:
            exclude_vids = set()
        self._prev_regular_clusters = [
            cl for cl in self.clusters.values() if cl.cluster_type == "regular"]
        for cid in list(self.clusters.keys()):
            if self.clusters[cid].cluster_type == "regular":
                del self.clusters[cid]

        available = []
        for vid, v in vehicles.items():
            if not self.in_coverage(v) or v.is_emergency:
                continue
            if vid in exclude_vids:
                continue
            if allowed_vids is not None and vid not in allowed_vids:
                continue
            available.append(v)

        unassigned = set(v.veh_id for v in available)
        while len(unassigned) >= REGULAR_CLUSTER_MIN_MEMBERS:
            best_vid = None
            best_count = -1
            for vid in unassigned:
                v = vehicles[vid]
                count = sum(
                    1 for oid in unassigned if oid != vid and
                    euclidean_distance_pos(v.position, vehicles[oid].position)
                    <= REGULAR_CLUSTER_RADIUS)
                if count > best_count:
                    best_count = count
                    best_vid = vid
            if best_vid is None or best_count < REGULAR_CLUSTER_MIN_MEMBERS - 1:
                break

            ch_v = vehicles[best_vid]
            member_list = [best_vid]
            unassigned.discard(best_vid)
            for vid in list(unassigned):
                v = vehicles[vid]
                if (euclidean_distance_pos(ch_v.position, v.position)
                        <= REGULAR_CLUSTER_RADIUS and v.dir_n == ch_v.dir_n):
                    member_list.append(vid)
                    unassigned.discard(vid)

            member_set = set(member_list)
            inherited_id = None
            best_jac = 0.0
            for prev_cl in self._prev_regular_clusters:
                inter = len(member_set & prev_cl.core_members)
                union = len(member_set | prev_cl.core_members)
                if union > 0:
                    jac = inter / union
                    if jac > best_jac:
                        best_jac = jac
                        inherited_id = prev_cl.cluster_id
            if inherited_id is not None and best_jac >= 0.5:
                cid = inherited_id
                self._prev_regular_clusters = [
                    c for c in self._prev_regular_clusters if c.cluster_id != cid]
            else:
                self._regular_cluster_counter += 1
                cid = f"{self.rsu_id}:RC{self._regular_cluster_counter}"

            new_cl = Cluster(cluster_id=cid, cluster_type="regular",
                             center=Position(ch_v.position.x, ch_v.position.y),
                             radius=REGULAR_CLUSTER_RADIUS,
                             ch_id=best_vid, creation_time=current_time)
            for vid in member_list:
                new_cl.core_members.add(vid)
                vehicles[vid].cluster_id = cid
                vehicles[vid].rsu_id = self.rsu_id
            self.clusters[cid] = new_cl

        for vid in unassigned:
            v = vehicles[vid]
            v.cluster_id = None
            v.mode = VehicleMode.UM
            v.connected_ch_id = None

    # ---------- RCH: nearest-to-BS rule (no scoring, no gate) ----------
    def select_relay_for_emergency_cluster(self, ec, vehicles, dsrc_v2v_range):
        ambulance = vehicles.get(ec.ch_id)
        if ambulance is None or BS_POSITION is None:
            ec.relay_ch_id = None
            return None

        best_vid = None
        best_dist = float("inf")
        for vid in ec.core_members:
            if vid == ec.ch_id or vid not in vehicles:
                continue
            d = euclidean_distance_pos(vehicles[vid].position, BS_POSITION)
            if d < best_dist:
                best_dist = d
                best_vid = vid

        ec.relay_ch_id = best_vid
        return best_vid

    # ---------- CH / backup election ----------
    def select_ch_and_backup_for_cluster(self, cluster_id, vehicles,
                                         dsrc_v2v_range, current_time):
        if cluster_id not in self.clusters:
            return None, None
        cl = self.clusters[cluster_id]
        all_members = list(cl.core_members | cl.guard_members)
        total_members = len(all_members)

        # Emergency cluster: CH = ambulance (fixed), everyone else CM/UM
        if cl.cluster_type == "emergency":
            if cl.ch_id and cl.ch_id in vehicles:
                for vid in all_members:
                    if vid not in vehicles:
                        continue
                    v = vehicles[vid]
                    if vid == cl.ch_id:
                        v.mode = VehicleMode.CH
                        v.connected_ch_id = cl.ch_id
                    elif vid == cl.relay_ch_id:
                        v.mode = VehicleMode.RCH
                        v.connected_ch_id = cl.ch_id
                    else:
                        amb = vehicles[cl.ch_id]
                        if euclidean_distance_pos(v.position, amb.position) <= dsrc_v2v_range:
                            v.mode = VehicleMode.CM
                            v.connected_ch_id = cl.ch_id
                        else:
                            v.mode = VehicleMode.UM
                            v.connected_ch_id = None
                return cl.ch_id, cl.relay_ch_id
            return None, None

        # Regular cluster
        if total_members < REGULAR_CLUSTER_MIN_MEMBERS:
            for vid in all_members:
                if vid in vehicles:
                    vehicles[vid].mode = VehicleMode.UM
                    vehicles[vid].connected_ch_id = None
                    vehicles[vid].cluster_id = None
            del self.clusters[cluster_id]
            return None, None

        scores: Dict[str, int] = {}
        for vid in all_members:
            if vid not in vehicles:
                continue
            v = vehicles[vid]
            reachable = sum(
                1 for oid in all_members if oid != vid and oid in vehicles
                and euclidean_distance_pos(v.position, vehicles[oid].position)
                <= dsrc_v2v_range)
            scores[vid] = reachable
        if not scores:
            return None, None

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        new_ch = ranked[0][0]
        new_backup = ranked[1][0] if len(ranked) > 1 else None

        old_ch = cl.ch_id
        if old_ch != new_ch:
            if old_ch and old_ch in vehicles:
                vehicles[old_ch].accumulated_ch_stable_time += max(
                    0.0, current_time - vehicles[old_ch].ch_role_start_time)
            if new_ch in vehicles:
                nv = vehicles[new_ch]
                nv.ch_change_count += 1
                nv.last_ch_change_time = current_time
                nv.ch_role_start_time = current_time
                nv.previous_ch_id = old_ch

        cl.ch_id = new_ch
        cl.relay_ch_id = new_backup
        for vid in all_members:
            if vid not in vehicles:
                continue
            v = vehicles[vid]
            if vid == new_ch:
                v.mode = VehicleMode.CH
                v.connected_ch_id = new_ch
            else:
                ch = vehicles[new_ch]
                if euclidean_distance_pos(v.position, ch.position) <= dsrc_v2v_range:
                    v.mode = VehicleMode.CM
                    v.connected_ch_id = new_ch
                else:
                    v.mode = VehicleMode.UM
                    v.connected_ch_id = None
        return new_ch, new_backup

    # ---------- Cross-cluster: disabled ----------
    def cross_cluster_optimization(self, vehicles, dsrc_v2v_range):
        # naive_ea does not perform cross-cluster reassignment
        pass


# =========================================================
# ClusteringManager (same API as ECDC)
# =========================================================
class ClusteringManager:
    def __init__(self, dsrc_v2v_range, delta_t, vehicle_length_l, A, B):
        if abs((A + B) - 1.0) > 1e-9:
            raise ValueError("A + B must be 1.")
        self.dsrc_v2v_range = dsrc_v2v_range
        self.delta_t = delta_t
        self.vehicle_length_l = vehicle_length_l
        self.A = A
        self.B = B
        self.vehicles: Dict[str, VehicleState] = {}
        self.rsus: Dict[str, RSU] = {}
        self.current_time: float = 0.0

    def add_vehicle(self, vehicle):
        vehicle.update_direction()
        self.vehicles[vehicle.veh_id] = vehicle

    def add_rsu(self, rsu):
        self.rsus[rsu.rsu_id] = rsu

    def run(self):
        self.current_time += self.delta_t
        for rsu in self.rsus.values():
            rsu.received_hello_msgs.clear()
        for v in self.vehicles.values():
            v.update_motion_state(self.delta_t)
            v.neighbor_count = 0
        for v in self.vehicles.values():
            for rsu in self.rsus.values():
                if rsu.in_coverage(v):
                    rsu.receive_hello_msg(v.build_hello_msg())
        ids = list(self.vehicles.keys())
        for v in self.vehicles.values():
            v.one_hop_neighbors.clear()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                n = self.vehicles[ids[i]]
                m = self.vehicles[ids[j]]
                if euclidean_distance_pos(n.position, m.position) <= self.dsrc_v2v_range \
                        and n.dir_n == m.dir_n:
                    n.one_hop_neighbors.add(m.veh_id)
                    m.one_hop_neighbors.add(n.veh_id)
        for v in self.vehicles.values():
            v.neighbor_count = len(v.one_hop_neighbors)

        self._update_clusters()
        return self._execute_ch_selection()

    def _update_clusters(self):
        ambulance = self.vehicles.get("ambulance")
        nearest_rsu_of = {}
        for vid, v in self.vehicles.items():
            covering = [r for r in self.rsus.values() if r.in_coverage(v)]
            if covering:
                nearest = min(covering, key=lambda r: euclidean_distance_pos(r.position, v.position))
                nearest_rsu_of[vid] = nearest.rsu_id

        owner_rsu = None
        if ambulance is not None:
            covering = [r for r in self.rsus.values() if r.in_coverage(ambulance)]
            if covering:
                owner_rsu = min(covering, key=lambda r: euclidean_distance_pos(r.position, ambulance.position))
        if owner_rsu is None and ambulance is not None:
            ambulance.cluster_id = None
            ambulance.rsu_id = None

        ec_member_vids: Set[str] = set()
        for rsu in self.rsus.values():
            if rsu is owner_rsu and ambulance is not None:
                ec = rsu.update_emergency_cluster(ambulance, self.vehicles, self.current_time)
                if ec is not None:
                    ec_member_vids = set(ec.core_members | ec.guard_members)
            else:
                for cid in list(rsu.clusters.keys()):
                    if rsu.clusters[cid].cluster_type == "emergency":
                        del rsu.clusters[cid]

        for rsu in self.rsus.values():
            allowed = {vid for vid, rid in nearest_rsu_of.items() if rid == rsu.rsu_id}
            rsu.update_regular_clusters(self.vehicles, self.current_time,
                                        exclude_vids=ec_member_vids,
                                        allowed_vids=allowed)

    def _execute_ch_selection(self):
        results = {}
        for rsu in self.rsus.values():
            for cid in list(rsu.clusters.keys()):
                if cid not in rsu.clusters:
                    continue
                cl = rsu.clusters[cid]
                ch_id, backup_id = rsu.select_ch_and_backup_for_cluster(
                    cluster_id=cid, vehicles=self.vehicles,
                    dsrc_v2v_range=self.dsrc_v2v_range,
                    current_time=self.current_time)
                if cl.cluster_type == "emergency":
                    rsu.select_relay_for_emergency_cluster(cl, self.vehicles, self.dsrc_v2v_range)
                results[cid] = {"primary_ch": cl.ch_id, "backup_ch": cl.relay_ch_id}
            rsu.cross_cluster_optimization(self.vehicles, self.dsrc_v2v_range)
        return results