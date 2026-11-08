
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# =========================================================
# Utilities
# =========================================================
def euclidean_distance_pos(a: "Position", b: "Position") -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def calculate_direction_angle_and_type(
    current_x: float,
    current_y: float,
    destination_x: float,
    destination_y: float
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

def along_distance(
    position: Position,
    start_pos: Position,
    end_pos: Position
) -> float:
    """Project position onto the line segment and return distance from start."""
    dx = end_pos.x - start_pos.x
    dy = end_pos.y - start_pos.y
    length_sq = dx*dx + dy*dy
    if length_sq <= 1e-12:
        return 0.0
    t = ((position.x - start_pos.x) * dx + (position.y - start_pos.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return t * math.sqrt(length_sq)

# =========================================================
# Enums and messages
# =========================================================
class VehicleMode(Enum):
    UM = "UM"
    CHA = "CHA"
    CH = "CH"
    CM = "CM"

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

@dataclass
class HelloNeighborMessage:
    veh_id: str
    lane_id: str
    edge_id: str
    x_n: float
    y_n: float
    v_n: float
    dir_n: int
    neighbors: List[str]

# =========================================================
# State classes
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
    zone_id: Optional[str] = None
    connected_ch_id: Optional[str] = None

    one_hop_neighbors: Set[str] = field(default_factory=set)

    acceleration: float = 0.0
    angle: float = 0.0
    nearest_rsu_id: Optional[str] = None
    sh_avg: float = 0.0
    v_n_value: float = 0.0
    neighbor_count: int = 0
    is_dead_member: bool = False

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
        zone_id: Optional[str] = None,
        connected_ch_id: Optional[str] = None
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
        self.zone_id = zone_id
        self.connected_ch_id = connected_ch_id
        self.one_hop_neighbors = set()
        self.acceleration = 0.0
        self.angle = 0.0
        self.nearest_rsu_id = None
        self.sh_avg = 0.0
        self.v_n_value = 0.0
        self.neighbor_count = 0
        self.is_dead_member = False
        self.update_direction()

    def update_direction(self) -> None:
        self.theta_n, self.dir_n = calculate_direction_angle_and_type(
            self.position.x, self.position.y,
            self.destination.x, self.destination.y
        )

    def build_hello_msg(self) -> HelloMessage:
        return HelloMessage(
            veh_id=self.veh_id,
            lane_id=self.lane_id,
            edge_id=self.edge_id,
            x_n=self.position.x,
            y_n=self.position.y,
            v_n=self.speed,
            dir_n=self.dir_n
        )

    def build_hello_neighbor_msg(self) -> HelloNeighborMessage:
        return HelloNeighborMessage(
            veh_id=self.veh_id,
            lane_id=self.lane_id,
            edge_id=self.edge_id,
            x_n=self.position.x,
            y_n=self.position.y,
            v_n=self.speed,
            dir_n=self.dir_n,
            neighbors=sorted(list(self.one_hop_neighbors))
        )

# =========================================================
# Zone (modified for along‑lane distances)
# =========================================================
@dataclass
class Zone:
    zone_id: str
    lane_id: str
    center: Position
    start_dist: float   # distance from lane start
    end_dist: float     # distance from lane start

    active_members: Set[str] = field(default_factory=set)
    dead_members: Set[str] = field(default_factory=set)
    ch_id: Optional[str] = None

    @property
    def length(self) -> float:
        return abs(self.end_dist - self.start_dist)

    def contains_dist(self, dist: float) -> bool:
        return self.start_dist <= dist <= self.end_dist

# =========================================================
# RSU (modified to store lane geometries and use along‑lane distance)
# =========================================================
class RSU:
    def __init__(
        self,
        rsu_id: str,
        position: Position,
        coverage_range: float = 900.0,
        coverage_radius: Optional[float] = None,
        hello_interval: float = 1.0
    ) -> None:
        self.rsu_id = rsu_id
        self.position = position
        self.coverage_range = coverage_radius if coverage_radius is not None else coverage_range
        self.coverage_radius = self.coverage_range
        self.hello_interval = hello_interval

        self.zones: Dict[str, Zone] = {}
        self.received_hello_msgs: Dict[str, HelloMessage] = {}
        self.received_hello_neighbor_msgs: Dict[str, HelloNeighborMessage] = {}

        # NEW: store lane geometries (start and end positions)
        self.lane_geometries: Dict[str, Tuple[Position, Position]] = {}

    def in_coverage(self, vehicle: VehicleState) -> bool:
        return euclidean_distance_pos(self.position, vehicle.position) <= self.coverage_range

    def set_lane_geometry(self, lane_id: str, start_pos: Position, end_pos: Position) -> None:
        """Register the shape of a lane (used for along‑lane distance calculations)."""
        self.lane_geometries[lane_id] = (start_pos, end_pos)

    def _get_along_dist(self, position: Position, lane_id: str) -> float:
        """Return the distance of a position from the lane start (projected)."""
        if lane_id not in self.lane_geometries:
            return 0.0
        start_pos, end_pos = self.lane_geometries[lane_id]
        return along_distance(position, start_pos, end_pos)

    def configure_lane_zones(self, lane_id: str) -> None:
        """
        Split the lane into 2 equal zones.
        The lane geometry must have been set via set_lane_geometry() first.
        """
        if lane_id not in self.lane_geometries:
            raise ValueError(f"Lane geometry not set for {lane_id}")
        start_pos, end_pos = self.lane_geometries[lane_id]
        lane_len = euclidean_distance_pos(start_pos, end_pos)
        if lane_len <= 0:
            return

        zone_len = lane_len / 3.0
        for i in range(3):
            center_dist = (i + 0.5) * zone_len
            t = center_dist / lane_len
            cx = start_pos.x + (end_pos.x - start_pos.x) * t
            cy = start_pos.y + (end_pos.y - start_pos.y) * t
            center = Position(cx, cy)

            zone_id = f"{self.rsu_id}:{lane_id}:ZONE_{i + 1}"
            start_dist = i * zone_len
            end_dist = (i + 1) * zone_len
            self.zones[zone_id] = Zone(
                zone_id=zone_id,
                lane_id=lane_id,
                center=center,
                start_dist=start_dist,
                end_dist=end_dist
            )

    def receive_hello_msg(self, msg: HelloMessage) -> None:
        self.received_hello_msgs[msg.veh_id] = msg

    def receive_hello_neighbor_msg(self, msg: HelloNeighborMessage) -> None:
        self.received_hello_neighbor_msgs[msg.veh_id] = msg

    def assign_vehicle_to_nearest_zone(self, vehicle: VehicleState) -> Optional[str]:
        candidate_zones = [z for z in self.zones.values() if z.lane_id == vehicle.lane_id]
        if not candidate_zones:
            return None

        # Clear old assignment
        for z in candidate_zones:
            z.active_members.discard(vehicle.veh_id)
            z.dead_members.discard(vehicle.veh_id)

        # Compute vehicle's along‑lane distance
        veh_dist = self._get_along_dist(vehicle.position, vehicle.lane_id)

        # Find zone that contains this distance
        containing = [z for z in candidate_zones if z.contains_dist(veh_dist)]
        if containing:
            selected = containing[0]
        else:
            # pick the zone whose centre is closest (in along‑lane distance)
            selected = min(candidate_zones, key=lambda z: abs(z.start_dist + z.length/2 - veh_dist))

        vehicle.zone_id = selected.zone_id
        vehicle.rsu_id = self.rsu_id
        return selected.zone_id

    def compute_lane_average_speed(
        self,
        lane_id: str,
        vehicles: Dict[str, VehicleState]
    ) -> float:
        speeds = [
            abs(v.speed)
            for v in vehicles.values()
            if v.lane_id == lane_id and self.in_coverage(v)
        ]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def compute_dead_zone_size(
        self,
        zone: Zone,
        vehicles: Dict[str, VehicleState]
    ) -> float:
        if zone.length <= 0.0:
            return 0.0
        lane_avg_v = self.compute_lane_average_speed(zone.lane_id, vehicles)
        dz = lane_avg_v * self.hello_interval
        return min(dz, zone.length / 2.0)

    def is_vehicle_in_dead_zone(
        self,
        vehicle: VehicleState,
        zone: Zone,
        vehicles: Dict[str, VehicleState]
    ) -> bool:
        dsz = self.compute_dead_zone_size(zone, vehicles)
        if dsz <= 0.0:
            return False
        veh_dist = self._get_along_dist(vehicle.position, vehicle.lane_id)
        dead_left = zone.start_dist <= veh_dist <= zone.start_dist + dsz
        dead_right = zone.end_dist - dsz <= veh_dist <= zone.end_dist
        return dead_left or dead_right

    def classify_zone_members_into_active_and_dead(
        self,
        vehicles: Dict[str, VehicleState]
    ) -> None:
        for z in self.zones.values():
            z.active_members.clear()
            z.dead_members.clear()

        for z in self.zones.values():
            members = [v for v in vehicles.values() if v.zone_id == z.zone_id and v.rsu_id == self.rsu_id]
            for v in members:
                if self.is_vehicle_in_dead_zone(v, z, vehicles):
                    z.dead_members.add(v.veh_id)
                else:
                    z.active_members.add(v.veh_id)

    def calculate_sh_avg(
        self,
        target_vehicle_id: str,
        vehicles: Dict[str, VehicleState],
        dsrc_v2v_range: float,
    ) -> float:
        """
        Compute average space headway using along‑lane distances.
        """
        if target_vehicle_id not in vehicles:
            return 0.0

        tv = vehicles[target_vehicle_id]
        tv_dist = self._get_along_dist(tv.position, tv.lane_id)

        front_gaps = []
        back_gaps = []

        for oid, ov in vehicles.items():
            if oid == target_vehicle_id:
                continue
            if ov.lane_id != tv.lane_id or ov.dir_n != tv.dir_n:
                continue
            dist = euclidean_distance_pos(tv.position, ov.position)
            if dist > dsrc_v2v_range:
                continue

            ov_dist = self._get_along_dist(ov.position, ov.lane_id)
            gap = abs(ov_dist - tv_dist)

            # Determine if ahead or behind based on along‑lane distance
            if ov_dist > tv_dist:
                front_gaps.append(gap)
            elif ov_dist < tv_dist:
                back_gaps.append(gap)

        gaps = []
        if front_gaps:
            gaps.append(min(front_gaps))
        if back_gaps:
            gaps.append(min(back_gaps))

        if not gaps:
            return 0.0
        return sum(gaps) / len(gaps)

    def calculate_v_n(
        self,
        target_vehicle_id: str,
        vehicles: Dict[str, VehicleState],
        delta_t: float,
        vehicle_length_l: float,
        dsrc_v2v_range: float
    ) -> float:
        if delta_t <= 0.0:
            return 0.0
        sh = self.calculate_sh_avg(target_vehicle_id, vehicles, dsrc_v2v_range)
        return (sh - vehicle_length_l) / delta_t

    def select_ch_for_zone(
            self,
            zone_id: str,
            vehicles: Dict[str, VehicleState],
            A: float,
            B: float,
            delta_t: float,
            vehicle_length_l: float,
            dsrc_v2v_range: float
    ) -> Optional[str]:
        if abs((A + B) - 1.0) > 1e-9:
            raise ValueError("A + B must be equal to 1.")

        z = self.zones[zone_id]
        all_members = list(z.active_members | z.dead_members)
        total_members = len(all_members)

        if total_members == 0:
            z.ch_id = None
            return None

        if total_members < 3:
            z.ch_id = None
            for vid in all_members:
                if vid in vehicles:
                    vehicles[vid].mode = VehicleMode.UM
                    vehicles[vid].connected_ch_id = None
            return None

        # ---- CH candidates: only vehicles in the active zone (paper, Algorithm 2) ----
        candidates = list(z.active_members)
        if not candidates:
            # No active vehicles – dissolve the cluster
            z.ch_id = None
            for vid in all_members:
                if vid in vehicles:
                    vehicles[vid].mode = VehicleMode.UM
                    vehicles[vid].connected_ch_id = None
            return None

        # Compute RM for candidates only
        v_n = {}
        l_cn = {}
        for vid in candidates:
            sh_avg = self.calculate_sh_avg(vid, vehicles, dsrc_v2v_range)
            v_n[vid] = (sh_avg - vehicle_length_l) / delta_t if delta_t > 0 else 0.0
            l_cn[vid] = euclidean_distance_pos(vehicles[vid].position, z.center)

        max_v = max(abs(v) for v in v_n.values()) if v_n else 0.0
        max_l = max(l_cn.values()) if l_cn else 0.0

        rm = {}
        for vid in candidates:
            norm_v = abs(v_n[vid]) / max_v if max_v != 0 else 0.0
            norm_l = l_cn[vid] / max_l if max_l != 0 else 0.0
            rm[vid] = A * norm_v + B * norm_l

        chosen = min(rm, key=rm.get)

        # Assign CH
        z.ch_id = chosen
        ch = vehicles[chosen]
        ch.mode = VehicleMode.CH
        ch.connected_ch_id = chosen
        ch.zone_id = z.zone_id
        ch.rsu_id = self.rsu_id

        # Assign modes to ALL members (active + dead)
        for vid in all_members:
            if vid == chosen:
                continue
            v = vehicles[vid]
            d2ch = euclidean_distance_pos(v.position, ch.position)
            if d2ch <= dsrc_v2v_range:
                v.mode = VehicleMode.CM
                v.connected_ch_id = chosen
                v.zone_id = z.zone_id
                v.rsu_id = self.rsu_id
            else:
                v.mode = VehicleMode.UM
                v.connected_ch_id = None

        return chosen

# =========================================================
# Clustering Manager (unchanged – it calls the RSU methods)
# =========================================================
class ClusteringManager:
    def __init__(
        self,
        dsrc_v2v_range: float,
        delta_t: float,
        vehicle_length_l: float,
        A: float,
        B: float
    ) -> None:
        if abs((A + B) - 1.0) > 1e-9:
            raise ValueError("A + B must be 1 according to the article.")
        self.dsrc_v2v_range = dsrc_v2v_range
        self.delta_t = delta_t
        self.vehicle_length_l = vehicle_length_l
        self.A = A
        self.B = B
        self.vehicles: Dict[str, VehicleState] = {}
        self.rsus: Dict[str, RSU] = {}

    def add_vehicle(self, vehicle: VehicleState) -> None:
        vehicle.update_direction()
        self.vehicles[vehicle.veh_id] = vehicle

    def add_rsu(self, rsu: RSU) -> None:
        self.rsus[rsu.rsu_id] = rsu

    def run(self) -> Dict[str, Optional[str]]:
        self.clear_rsu_buffers()
        self.set_all_vehicles_to_cha()
        self.broadcast_hello_msg()
        self.build_neighbors_from_received_hello()
        self.assign_covered_vehicles_to_nearest_zone()
        self.send_hello_neigh_to_rsus()
        return self.execute_ch_selection()

    def clear_rsu_buffers(self) -> None:
        for rsu in self.rsus.values():
            rsu.received_hello_msgs.clear()
            rsu.received_hello_neighbor_msgs.clear()

    def set_all_vehicles_to_cha(self) -> None:
        for v in self.vehicles.values():
            v.mode = VehicleMode.CHA
            v.connected_ch_id = None
            v.rsu_id = None
            v.zone_id = None
            v.sh_avg = 0.0
            v.v_n_value = 0.0
            v.neighbor_count = 0
            v.is_dead_member = False
            v.update_direction()

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
                if d <= self.dsrc_v2v_range and n.lane_id == m.lane_id and n.dir_n == m.dir_n:
                    n.one_hop_neighbors.add(m.veh_id)
                    m.one_hop_neighbors.add(n.veh_id)

        for v in self.vehicles.values():
            v.neighbor_count = len(v.one_hop_neighbors)

    def assign_covered_vehicles_to_nearest_zone(self) -> None:
        for v in self.vehicles.values():
            nearest_rsu = self.find_covering_rsu_for_vehicle(v)
            if nearest_rsu:
                nearest_rsu.assign_vehicle_to_nearest_zone(v)

        for rsu in self.rsus.values():
            rsu.classify_zone_members_into_active_and_dead(self.vehicles)
            for zone in rsu.zones.values():
                for vid in zone.active_members:
                    if vid in self.vehicles:
                        self.vehicles[vid].is_dead_member = False
                for vid in zone.dead_members:
                    if vid in self.vehicles:
                        self.vehicles[vid].is_dead_member = True

    def find_covering_rsu_for_vehicle(self, vehicle: VehicleState) -> Optional[RSU]:
        candidates = [r for r in self.rsus.values() if r.in_coverage(vehicle)]
        if not candidates:
            return None
        return min(candidates, key=lambda r: euclidean_distance_pos(r.position, vehicle.position))

    def send_hello_neigh_to_rsus(self) -> None:
        for v in self.vehicles.values():
            if not v.rsu_id or v.rsu_id not in self.rsus:
                continue
            rsu = self.rsus[v.rsu_id]
            msg = v.build_hello_neighbor_msg()
            rsu.receive_hello_neighbor_msg(msg)

    def execute_ch_selection(self) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        for rsu in self.rsus.values():
            for zid in rsu.zones.keys():
                sel = rsu.select_ch_for_zone(
                    zone_id=zid,
                    vehicles=self.vehicles,
                    A=self.A,
                    B=self.B,
                    delta_t=self.delta_t,
                    vehicle_length_l=self.vehicle_length_l,
                    dsrc_v2v_range=self.dsrc_v2v_range
                )
                results[zid] = sel
        return results