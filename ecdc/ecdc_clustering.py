
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# =========================================================
# ECDC Tuning Parameters
# =========================================================
EMERGENCY_CLUSTER_BASE_RADIUS   = 100.0   # m
EMERGENCY_CLUSTER_LOOKAHEAD     = 10.0     # s (multiplied by ambulance speed)
EMERGENCY_CLUSTER_MIN_RADIUS    = 50.0    # m
EMERGENCY_CLUSTER_MAX_RADIUS    = 400.0   # m (= comm range)

REGULAR_CLUSTER_RADIUS          = 150.0   # m
REGULAR_CLUSTER_MIN_MEMBERS     = 3

HANDOVER_HYSTERESIS             = 0.15    # 15% improvement

MAX_LINK_LIFETIME               = 10.0

# Relay CH scoring weights (sum = 1.0)
WEIGHT_LINK_TO_EV               = 0.35
WEIGHT_LINK_TO_BS               = 0.25
WEIGHT_POSITION_AHEAD           = 0.20
WEIGHT_SPEED_SYNC               = 0.10
WEIGHT_ACCEL_STABILITY          = 0.10




# =========================================================
# Ablation mode (set via ECDC_ABLATION env var)
# =========================================================
import os as _os
ABLATION_MODE = _os.environ.get('ECDC_ABLATION', '').strip().lower()
#   'directbs' is handled in the harness (forces direct-to-BS for ambulance).

# Relay gate fraction (gamma)
RELAY_GATE_FRACTION = 0.5


# =========================================================
# Helpers
# =========================================================
def euclidean_distance_pos(a: "Position", b: "Position") -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def calculate_direction_angle_and_type(
    current_x: float, current_y: float,
    destination_x: float, destination_y: float
) -> Tuple[float, int]:
    dx = destination_x - current_x
    dy = destination_y - current_y
    theta = math.degrees(math.atan2(dy, dx))
    while theta < 0:
        theta += 360.0
    while theta >= 360.0:
        theta -= 360.0
    if 0.0 <= theta < 90.0:
        dir_type = 1
    elif 90.0 <= theta < 180.0:
        dir_type = 2
    elif 180.0 <= theta < 270.0:
        dir_type = 3
    else:
        dir_type = 4
    return theta, dir_type


def predict_link_lifetime(
    pos1: "Position", speed1: float,
    pos2: "Position", speed2: float,
    comm_range: float
) -> float:
    """
    Predict remaining link lifetime between two vehicles.
    Uses the difference in scalar speed as the closing/opening rate.
    """
    dist = euclidean_distance_pos(pos1, pos2)
    if dist >= comm_range:
        return 0.0
    rel_speed = abs(speed1 - speed2)
    if rel_speed < 0.1:
        return MAX_LINK_LIFETIME
    return min((comm_range - dist) / rel_speed, MAX_LINK_LIFETIME)


# =========================================================
# Enums and Messages
# =========================================================
class VehicleMode(Enum):
    UM  = "UM"    # Unclustered Member
    CH  = "CH"    # Cluster Head
    CM  = "CM"    # Cluster Member
    RCH = "RCH"   # Relay Cluster Head (emergency-specific)


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
# VehicleState
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

    # Legacy compatibility fields (unused by ECDC but kept for harness)
    sh_avg: float = 0.0
    v_n_value: float = 0.0
    is_dead_member: bool = False
    predicted_ert: float = 0.0

    def __init__(
        self,
        veh_id: Optional[str] = None,
        lane_id: str = "",
        edge_id: str = "",
        position: Optional[Position] = None,
        speed: float = 0.0,
        destination: Optional[Position] = None,
        theta_n: float = 0.0,
        dir_n: int = 1,
        mode: VehicleMode = VehicleMode.UM,
        rsu_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        connected_ch_id: Optional[str] = None,
        is_emergency: bool = False,
    ) -> None:
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

    def update_direction(self) -> None:
        self.theta_n, self.dir_n = calculate_direction_angle_and_type(
            self.position.x, self.position.y,
            self.destination.x, self.destination.y
        )

    def update_motion_state(self, delta_t: float) -> None:
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

    def build_hello_msg(self) -> HelloMessage:
        return HelloMessage(
            veh_id=self.veh_id, lane_id=self.lane_id, edge_id=self.edge_id,
            x_n=self.position.x, y_n=self.position.y, v_n=self.speed,
            dir_n=self.dir_n, is_emergency=self.is_emergency
        )


# =========================================================
# Cluster (replaces Zone)
# =========================================================
@dataclass
class Cluster:
    cluster_id: str
    cluster_type: str                      # "emergency" or "regular"
    center: Position
    radius: float
    ch_id: Optional[str] = None
    relay_ch_id: Optional[str] = None      # emergency only
    core_members: Set[str] = field(default_factory=set)
    guard_members: Set[str] = field(default_factory=set)
    structural_active_time: float = 0.0
    creation_time: float = 0.0

    @property
    def length(self) -> float:
        return self.radius * 2.0

    def contains_point(self, point: Position) -> bool:
        return euclidean_distance_pos(point, self.center) <= self.radius

    def contains_x(self, x: float) -> bool:
        # Legacy compatibility; not used for emergency clusters
        return False


# =========================================================
# ECDC RSU
# =========================================================
class RSU:
    def __init__(
        self,
        rsu_id: str,
        position: Position,
        coverage_range: float = 900.0,
        coverage_radius: Optional[float] = None,
        hello_interval: float = 1.0,
        hysteresis_threshold: float = 0.3,
        min_dwell_time: float = 4.0,
        ert_handover_threshold: float = 3.0,
        guard_ratio: float = 0.2,
        overlap_ratio: float = 0.15,
    ) -> None:
        self.rsu_id = rsu_id
        self.position = position
        self.coverage_range = coverage_radius if coverage_radius is not None else coverage_range
        self.coverage_radius = self.coverage_range
        self.hello_interval = hello_interval
        self.hysteresis_threshold = hysteresis_threshold
        self.min_dwell_time = min_dwell_time
        self.ert_handover_threshold = ert_handover_threshold
        self.guard_ratio = guard_ratio
        self.overlap_ratio = overlap_ratio

        self.clusters: Dict[str, Cluster] = {}
        self.received_hello_msgs: Dict[str, HelloMessage] = {}

        self._regular_cluster_counter = 0
        self._prev_regular_clusters: List[Cluster] = []

        # Legacy compatibility no-ops (harness may call them)
        self.lane_geometries: Dict[str, Tuple[Position, Position]] = {}

    # ---------- Legacy no-ops for harness compatibility ----------
    def set_lane_geometry(self, lane_id: str, start_pos: Position, end_pos: Position) -> None:
        self.lane_geometries[lane_id] = (start_pos, end_pos)

    def configure_lane_zones(self, lane_id: str, zone_centers: List[Position]) -> None:
        # ECDC does not use fixed zones. No-op.
        pass

    def configure_zones(self, zone_centers: List[Position], radius: Optional[float] = None) -> None:
        # ECDC does not use fixed zones. No-op.
        pass

    # ---------- Coverage ----------
    def in_coverage(self, vehicle: VehicleState) -> bool:
        return euclidean_distance_pos(self.position, vehicle.position) <= self.coverage_range

    # ---------- Messages ----------
    def receive_hello_msg(self, msg: HelloMessage) -> None:
        self.received_hello_msgs[msg.veh_id] = msg


    # =========================================================
    # Emergency cluster management
    # =========================================================
    def update_emergency_cluster(
        self,
        ambulance: VehicleState,
        vehicles: Dict[str, VehicleState],
        current_time: float,
    ) -> Optional[Cluster]:
        """
        Create or update the emergency cluster centred on the ambulance.
        The ambulance is always the CH. Radius grows with ambulance speed.
        """
        if not self.in_coverage(ambulance):
            for cid in list(self.clusters.keys()):
                if self.clusters[cid].cluster_type == "emergency":
                    del self.clusters[cid]
            return None

        # Dynamic radius (or fixed if ablation mode)
        if ABLATION_MODE == 'fixedradius':
            dynamic_radius = EMERGENCY_CLUSTER_BASE_RADIUS
        else:
            dynamic_radius = (
                    EMERGENCY_CLUSTER_BASE_RADIUS
                    + abs(ambulance.speed) * EMERGENCY_CLUSTER_LOOKAHEAD
            )
            dynamic_radius = clamp(
                dynamic_radius,
                EMERGENCY_CLUSTER_MIN_RADIUS,
                EMERGENCY_CLUSTER_MAX_RADIUS,
            )

        ec_id = f"{self.rsu_id}:EC"
        if ec_id not in self.clusters:
            self.clusters[ec_id] = Cluster(
                cluster_id=ec_id,
                cluster_type="emergency",
                center=Position(ambulance.position.x, ambulance.position.y),
                radius=dynamic_radius,
                ch_id=ambulance.veh_id,
                creation_time=current_time,
            )

        ec = self.clusters[ec_id]
        ec.center = Position(ambulance.position.x, ambulance.position.y)
        ec.radius = dynamic_radius
        ec.ch_id = ambulance.veh_id
        ec.core_members.clear()
        ec.guard_members.clear()
        ec.core_members.add(ambulance.veh_id)
        ambulance.cluster_id = ec_id
        ambulance.rsu_id = self.rsu_id

        # Populate members
        for vid, v in vehicles.items():
            if vid == ambulance.veh_id:
                continue
            if not self.in_coverage(v):
                continue
            if v.is_emergency:
                continue
            dist_to_amb = euclidean_distance_pos(v.position, ambulance.position)
            if dist_to_amb <= dynamic_radius and v.dir_n == ambulance.dir_n:
                ec.core_members.add(vid)
                v.cluster_id = ec_id
                v.rsu_id = self.rsu_id

        return ec

    # =========================================================
    # Regular clusters (greedy)
    # =========================================================
    def update_regular_clusters(
            self,
            vehicles: Dict[str, VehicleState],
            current_time: float,
            exclude_vids: Optional[Set[str]] = None,
            allowed_vids: Optional[Set[str]] = None
    ) -> None:
        if exclude_vids is None:
            exclude_vids = set()

        # Remember previous regular clusters for ID-inheritance via Jaccard overlap
        self._prev_regular_clusters = [
            cl for cl in self.clusters.values() if cl.cluster_type == "regular"
        ]

        # Tear down previous regular clusters
        for cid in list(self.clusters.keys()):
            if self.clusters[cid].cluster_type == "regular":
                del self.clusters[cid]

        # Candidates: in coverage, non-emergency, and not in ANY emergency cluster
        available: List[VehicleState] = []
        for vid, v in vehicles.items():
            if not self.in_coverage(v):
                continue
            if v.is_emergency:
                continue
            if vid in exclude_vids:  # ← global EC exclusion
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
                count = 0
                for other_id in unassigned:
                    if other_id == vid:
                        continue
                    if euclidean_distance_pos(v.position, vehicles[other_id].position) <= REGULAR_CLUSTER_RADIUS:
                        count += 1
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
                if (
                        euclidean_distance_pos(ch_v.position, v.position) <= REGULAR_CLUSTER_RADIUS
                        and v.dir_n == ch_v.dir_n
                ):
                    member_list.append(vid)
                    unassigned.discard(vid)

            # --- Stable-ID inheritance via Jaccard overlap on member set ---
            if ABLATION_MODE == 'nojaccard':
                # No identity inheritance: allocate a fresh monotonic ID
                # for every regular cluster at every step.
                self._regular_cluster_counter += 1
                cid = f"{self.rsu_id}:RC{self._regular_cluster_counter}"
            else:
                member_set = set(member_list)
                inherited_id = None
                best_jac = 0.0
                for prev_cl in self._prev_regular_clusters:
                    prev_set = prev_cl.core_members
                    inter = len(member_set & prev_set)
                    union = len(member_set | prev_set)
                    if union > 0:
                        jac = inter / union
                        if jac > best_jac:
                            best_jac = jac
                            inherited_id = prev_cl.cluster_id

                if inherited_id is not None and best_jac >= 0.5:
                    cid = inherited_id
                    # prevent reuse of the same previous cluster
                    self._prev_regular_clusters = [
                        c for c in self._prev_regular_clusters if c.cluster_id != cid
                    ]
                else:
                    self._regular_cluster_counter += 1
                    cid = f"{self.rsu_id}:RC{self._regular_cluster_counter}"

            new_cl = Cluster(
                cluster_id=cid,
                cluster_type="regular",
                center=Position(ch_v.position.x, ch_v.position.y),
                radius=REGULAR_CLUSTER_RADIUS,
                ch_id=best_vid,
                creation_time=current_time,
            )
            for vid in member_list:
                new_cl.core_members.add(vid)
                vehicles[vid].cluster_id = cid
                vehicles[vid].rsu_id = self.rsu_id

            self.clusters[cid] = new_cl

        # Clear leftovers that never made it into a cluster
        for vid in unassigned:
            v = vehicles[vid]
            v.cluster_id = None
            v.mode = VehicleMode.UM
            v.connected_ch_id = None

    # =========================================================
    # Relay CH selection (emergency cluster only)
    # =========================================================
    def select_relay_for_emergency_cluster(
        self,
        ec: Cluster,
        vehicles: Dict[str, VehicleState],
        dsrc_v2v_range: float,
    ) -> Optional[str]:
        ambulance = vehicles.get(ec.ch_id)
        if ambulance is None:
            ec.relay_ch_id = None
            return None

        # Relay gate : filter to feasible candidates BEFORE scoring.
        # Unless in 'nogate' ablation, RCH must lie within gamma * V2V range.
        if ABLATION_MODE in ('nogate', 'oldrch'):
            # 'nogate': no gate at all (selection or forwarding)
            # 'oldrch': selection over all EC members; forwarding gate still applied in harness
            gate_dist = dsrc_v2v_range
        else:
            gate_dist = RELAY_GATE_FRACTION * dsrc_v2v_range

        candidates = [
            vid for vid in ec.core_members
            if vid != ec.ch_id
               and vid in vehicles
               and euclidean_distance_pos(vehicles[vid].position,
                                          ambulance.position) <= gate_dist
        ]
        if not candidates:
            ec.relay_ch_id = None
            return None

        # Random-relay ablation: skip 5-factor scoring entirely.
        if ABLATION_MODE == 'randrelay':
            import random as _random
            ec.relay_ch_id = _random.choice(candidates)
            return ec.relay_ch_id

        best_vid = None
        best_score = -1.0

        for vid in candidates:
            v = vehicles[vid]

            # 1. Link lifetime to ambulance
            ll_amb = predict_link_lifetime(
                v.position, v.speed,
                ambulance.position, ambulance.speed,
                dsrc_v2v_range,
            )
            link_to_ev = clamp(ll_amb / MAX_LINK_LIFETIME, 0.0, 1.0)

            # 2. V2I RSSI (closeness to RSU)
            dist_rsu = euclidean_distance_pos(v.position, self.position)
            rssi = 1.0 - clamp(dist_rsu / self.coverage_range, 0.0, 1.0)

            # 3. Position ahead of the ambulance
            dx = v.position.x - ambulance.position.x
            dy = v.position.y - ambulance.position.y
            rad = math.radians(ambulance.angle)
            vx, vy = math.cos(rad), math.sin(rad)
            dot = dx * vx + dy * vy
            ahead = clamp(dot / dsrc_v2v_range + 0.5, 0.0, 1.0) if dot > 0 else 0.0

            # 4. Speed synchronisation with ambulance
            speed_diff = abs(abs(v.speed) - abs(ambulance.speed))
            speed_sync = 1.0 - clamp(speed_diff / 20.0, 0.0, 1.0)

            # 5. Acceleration stability
            accel_score = 1.0 - clamp(abs(v.acceleration) / 5.0, 0.0, 1.0)

            score = (
                WEIGHT_LINK_TO_EV     * link_to_ev +
                WEIGHT_LINK_TO_BS     * rssi +
                WEIGHT_POSITION_AHEAD * ahead +
                WEIGHT_SPEED_SYNC     * speed_sync +
                WEIGHT_ACCEL_STABILITY * accel_score
            )

            if score > best_score:
                best_score = score
                best_vid = vid

        ec.relay_ch_id = best_vid
        return best_vid


    # =========================================================
    # CH selection (called from harness)
    # =========================================================
    def select_ch_and_backup_for_cluster(
        self,
        cluster_id: str,
        vehicles: Dict[str, VehicleState],
        dsrc_v2v_range: float,
        current_time: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        if cluster_id not in self.clusters:
            return None, None

        cl = self.clusters[cluster_id]
        all_members = list(cl.core_members | cl.guard_members)
        total_members = len(all_members)

        # ---------- Emergency cluster ----------
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

        # ---------- Regular cluster ----------
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
                1 for oid in all_members
                if oid != vid and oid in vehicles
                and euclidean_distance_pos(v.position, vehicles[oid].position) <= dsrc_v2v_range
            )
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
                    0.0, current_time - vehicles[old_ch].ch_role_start_time
                )
            if new_ch in vehicles:
                nv = vehicles[new_ch]
                nv.ch_change_count += 1
                nv.last_ch_change_time = current_time
                nv.ch_role_start_time = current_time
                nv.previous_ch_id = old_ch

        cl.ch_id = new_ch
        cl.relay_ch_id = new_backup  # reuse field for backup in regular clusters

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

    # =========================================================
    # Cross-cluster optimization
    # =========================================================
    def cross_cluster_optimization(
        self,
        vehicles: Dict[str, VehicleState],
        dsrc_v2v_range: float,
    ) -> None:
        all_chs: List[Tuple[str, VehicleState]] = []
        for cl in self.clusters.values():
            if cl.ch_id and cl.ch_id in vehicles:
                all_chs.append((cl.ch_id, vehicles[cl.ch_id]))

        for cl in self.clusters.values():
            for vid in cl.core_members | cl.guard_members:
                if vid not in vehicles:
                    continue
                v = vehicles[vid]
                if v.mode in (VehicleMode.CH, VehicleMode.RCH):
                    continue

                current_ch_id = v.connected_ch_id
                current_dist = float("inf")
                if current_ch_id and current_ch_id in vehicles:
                    current_dist = euclidean_distance_pos(
                        v.position, vehicles[current_ch_id].position
                    )

                # Step 1: find the globally nearest eligible candidate
                # (no hysteresis inside the loop — this eliminates order dependence)
                best_ch_id = current_ch_id
                best_dist = current_dist

                for other_ch_id, other_ch in all_chs:
                    if other_ch_id == vid:
                        continue
                    d = euclidean_distance_pos(v.position, other_ch.position)
                    if d <= dsrc_v2v_range and d < best_dist:
                        best_dist = d
                        best_ch_id = other_ch_id

                # Step 2: apply hysteresis ONCE against the original serving distance
                hysteresis = 0.0 if ABLATION_MODE == 'nohyst' else HANDOVER_HYSTERESIS

                if best_ch_id != current_ch_id and best_dist < (1.0 - hysteresis) * current_dist:
                    v.connected_ch_id = best_ch_id
                    v.mode = VehicleMode.CM


# =========================================================
# ECDC Clustering Manager
# =========================================================
class ClusteringManager:
    def __init__(
        self,
        dsrc_v2v_range: float,
        delta_t: float,
        vehicle_length_l: float,
        A: float,
        B: float,
    ) -> None:
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

    def add_vehicle(self, vehicle: VehicleState) -> None:
        vehicle.update_direction()
        self.vehicles[vehicle.veh_id] = vehicle

    def add_rsu(self, rsu: RSU) -> None:
        self.rsus[rsu.rsu_id] = rsu

    def run(self) -> Dict[str, Dict[str, Optional[str]]]:
        self.current_time += self.delta_t
        self.clear_rsu_buffers()
        self.update_vehicle_motion_states()
        self.broadcast_hello_msg()
        self.build_neighbors_from_received_hello()
        self.update_clusters()
        result = self.execute_ch_selection_and_maintenance()
        return result

    # ---------- Internal steps ----------
    def clear_rsu_buffers(self) -> None:
        for rsu in self.rsus.values():
            rsu.received_hello_msgs.clear()

    def update_vehicle_motion_states(self) -> None:
        for v in self.vehicles.values():
            v.update_motion_state(self.delta_t)
            v.sh_avg = 0.0
            v.v_n_value = 0.0
            v.neighbor_count = 0
            v.is_dead_member = False

    def broadcast_hello_msg(self) -> None:
        for v in self.vehicles.values():
            msg = v.build_hello_msg()
            for rsu in self.rsus.values():
                if rsu.in_coverage(v):
                    rsu.receive_hello_msg(msg)

    def build_neighbors_from_received_hello(self) -> None:
        for v in self.vehicles.values():
            v.one_hop_neighbors.clear()
        ids = list(self.vehicles.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                n = self.vehicles[ids[i]]
                m = self.vehicles[ids[j]]
                d = euclidean_distance_pos(n.position, m.position)
                if d <= self.dsrc_v2v_range and n.dir_n == m.dir_n:
                    n.one_hop_neighbors.add(m.veh_id)
                    m.one_hop_neighbors.add(n.veh_id)
        for v in self.vehicles.values():
            v.neighbor_count = len(v.one_hop_neighbors)

    def find_covering_rsu_for_vehicle(self, vehicle: VehicleState) -> Optional[RSU]:
        candidates = [r for r in self.rsus.values() if r.in_coverage(vehicle)]
        if not candidates:
            return None
        return min(candidates, key=lambda r: euclidean_distance_pos(r.position, vehicle.position))

    def update_clusters(self) -> None:
        ambulance = self.vehicles.get("ambulance")

        # Nearest-RSU map (exclusive ownership of every vehicle)
        nearest_rsu_of: Dict[str, str] = {}
        for vid, v in self.vehicles.items():
            covering = [r for r in self.rsus.values() if r.in_coverage(v)]
            if covering:
                nearest = min(
                    covering,
                    key=lambda r: euclidean_distance_pos(r.position, v.position),
                )
                nearest_rsu_of[vid] = nearest.rsu_id

        # --- Owner RSU for the emergency cluster ---
        owner_rsu: Optional[RSU] = None
        if ambulance is not None:
            covering = [r for r in self.rsus.values() if r.in_coverage(ambulance)]
            if covering:
                owner_rsu = min(
                    covering,
                    key=lambda r: euclidean_distance_pos(r.position, ambulance.position),
                )

        if owner_rsu is None and ambulance is not None:
            ambulance.cluster_id = None
            ambulance.rsu_id = None

        ec_member_vids: Set[str] = set()

        for rsu in self.rsus.values():
            if rsu is owner_rsu:
                ec = rsu.update_emergency_cluster(ambulance, self.vehicles, self.current_time)
                if ec is not None:
                    ec_member_vids = set(ec.core_members | ec.guard_members)
            else:
                for cid in list(rsu.clusters.keys()):
                    if rsu.clusters[cid].cluster_type == "emergency":
                        del rsu.clusters[cid]

        # Regular clustering — one RSU owns each vehicle
        for rsu in self.rsus.values():
            allowed = {vid for vid, rid in nearest_rsu_of.items() if rid == rsu.rsu_id}
            rsu.update_regular_clusters(
                self.vehicles,
                self.current_time,
                exclude_vids=ec_member_vids,
                allowed_vids=allowed,
            )

    def execute_ch_selection_and_maintenance(self) -> Dict[str, Dict[str, Optional[str]]]:
        results: Dict[str, Dict[str, Optional[str]]] = {}
        for rsu in self.rsus.values():
            for cid in list(rsu.clusters.keys()):
                if cid not in rsu.clusters:
                    continue
                cl = rsu.clusters[cid]

                ch_id, backup_id = rsu.select_ch_and_backup_for_cluster(
                    cluster_id=cid,
                    vehicles=self.vehicles,
                    dsrc_v2v_range=self.dsrc_v2v_range,
                    current_time=self.current_time,
                )

                if cl.cluster_type == "emergency":
                    rsu.select_relay_for_emergency_cluster(
                        cl, self.vehicles, self.dsrc_v2v_range
                    )

                results[cid] = {
                    "primary_ch": cl.ch_id,
                    "backup_ch": cl.relay_ch_id,
                }

            # Cross-cluster optimization runs once per RSU, after all clusters
            # on this RSU have elected their CHs.
            rsu.cross_cluster_optimization(self.vehicles, self.dsrc_v2v_range)

        return results