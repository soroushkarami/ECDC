# MCSC_clustering.py
# Multi-criteria based stable clustering technique for VANETs
# Based on: "Multi-criteria based stable clustering technique for vehicular ad-hoc networks"
# Scientific Reports, Nature, 2026.

import math
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
COMM_RANGE = 250.0
CLS_THRESHOLD = 5.0
BEACON_TIMEOUT = 2.0
INF_CLS = 100.0

class VehicleStateEnum:
    UC = 0
    CH = 1
    CM = 2
    SCH = 3
    BN = 4

@dataclass
class VehicleState:
    veh_id: str
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0
    direction: float = 0.0
    lane_id: str = ""

    state: int = VehicleStateEnum.UC
    ch_id: Optional[str] = None
    cluster_id: Optional[str] = None
    ccf: float = 0.0
    avg_cls: float = 0.0
    deg_conn: int = 0
    avg_pcl: float = 0.0

    total_pcl: float = 0.0
    pcl_count: int = 0
    ch_role_start_time: float = 0.0

    stable_neighbors: Set[str] = field(default_factory=set)
    beacon_miss_count: int = 0

    def update_pcl(self, end_time: float) -> None:
        if self.ch_role_start_time > 0 and self.state == VehicleStateEnum.CH:
            duration = end_time - self.ch_role_start_time
            if duration > 0:
                self.total_pcl += duration
                self.pcl_count += 1
                self.avg_pcl = self.total_pcl / self.pcl_count if self.pcl_count > 0 else 0.0

    def reset_ch_timer(self, start_time: float) -> None:
        self.ch_role_start_time = start_time


class MCSCManager:
    def __init__(self, comm_range: float = COMM_RANGE, cls_threshold: float = CLS_THRESHOLD):
        self.comm_range = comm_range
        self.cls_threshold = cls_threshold
        self.vehicles: Dict[str, VehicleState] = {}
        self.current_time: float = 0.0
        self.next_cluster_id: int = 0
        self.ch_to_cluster: Dict[str, str] = {}   # CH vehicle ID -> persistent cluster ID

    def add_vehicle(self, veh_state: VehicleState) -> None:
        self.vehicles[veh_state.veh_id] = veh_state

    def process_step(self, current_time: float) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
        self.current_time = current_time

        self._compute_stable_neighbors()
        self._compute_ccf_all()
        self._elect_chs()
        self._assign_cms_and_schs()
        self._merge_clusters()
        self._maintain_clusters()

        clusters = {}
        ch_map = {}
        for vid, v in self.vehicles.items():
            if v.state == VehicleStateEnum.CH:
                cid = self._get_cluster_id(vid)
                if cid is None:
                    cid = self._assign_new_cluster_id(vid)
                clusters[cid] = set()
                ch_map[cid] = vid

        for vid, v in self.vehicles.items():
            if v.state in (VehicleStateEnum.CM, VehicleStateEnum.SCH) and v.ch_id is not None:
                cid = self._get_cluster_id(v.ch_id)
                if cid is not None and cid in clusters:
                    clusters[cid].add(vid)

        # Add CH itself to its cluster's member set
        for cid, ch_veh in ch_map.items():
            if cid in clusters:
                clusters[cid].add(ch_veh)

        return clusters, ch_map

    # ---------- helper cluster ID methods ----------
    def _assign_new_cluster_id(self, ch_id: str) -> str:
        self.next_cluster_id += 1
        cid = f"C_{self.next_cluster_id}"
        self.ch_to_cluster[ch_id] = cid
        if ch_id in self.vehicles:
            self.vehicles[ch_id].cluster_id = cid
        return cid

    def _get_cluster_id(self, ch_id: str) -> Optional[str]:
        return self.ch_to_cluster.get(ch_id)

    def _transfer_cluster_id(self, old_ch: str, new_ch: str) -> str:
        cid = self.ch_to_cluster.pop(old_ch, None)
        if cid is None:
            cid = self._assign_new_cluster_id(new_ch)
        else:
            self.ch_to_cluster[new_ch] = cid
            if new_ch in self.vehicles:
                self.vehicles[new_ch].cluster_id = cid
        return cid

    # ---------- stable neighbours (Algorithm 1) ----------
    def _compute_stable_neighbors(self) -> None:
        for vid, v in self.vehicles.items():
            v.stable_neighbors.clear()
            v.avg_cls = 0.0
            v.deg_conn = 0
            for oid, ov in self.vehicles.items():
                if oid == vid:
                    continue
                if not self._same_direction(v.direction, ov.direction):
                    continue
                dist = self._euclidean_dist(v.x, v.y, ov.x, ov.y)
                if dist > self.comm_range:
                    continue
                rel_speed = abs(v.speed - ov.speed)
                if rel_speed < 0.001:
                    cls = INF_CLS
                else:
                    cls = (self.comm_range - dist) / rel_speed
                    if cls < 0:
                        cls = 0.0
                if cls > self.cls_threshold:
                    v.stable_neighbors.add(oid)
                    v.avg_cls += cls
                    v.deg_conn += 1
            if v.deg_conn > 0:
                v.avg_cls = v.avg_cls / v.deg_conn

    # ---------- CCF (Eq. 2-5) ----------
    def _compute_ccf_all(self) -> None:
        for vid, v in self.vehicles.items():
            if v.deg_conn == 0 and v.avg_cls == 0 and v.avg_pcl == 0:
                v.ccf = 0.0
                continue
            total = v.avg_cls + v.deg_conn + v.avg_pcl
            if total == 0:
                v.ccf = 0.0
                continue
            alpha = v.avg_cls / total
            beta = v.deg_conn / total
            gamma = v.avg_pcl / total
            v.ccf = (alpha * v.avg_cls) + (beta * v.deg_conn) + (gamma * v.avg_pcl)

    # ---------- CH election ----------
    def _elect_chs(self) -> None:
        new_chs = set()
        for vid, v in self.vehicles.items():
            if v.deg_conn == 0:
                continue
            is_max = True
            for nid in v.stable_neighbors:
                nv = self.vehicles[nid]
                # Compare only with UC or CH candidates (not followers)
                if nv.state in (VehicleStateEnum.UC, VehicleStateEnum.CH):
                    if nv.ccf > v.ccf:
                        is_max = False
                        break
            if is_max:
                new_chs.add(vid)

        for vid, v in self.vehicles.items():
            if vid in new_chs:
                if v.state != VehicleStateEnum.CH:
                    v.state = VehicleStateEnum.CH
                    v.ch_id = vid
                    v.reset_ch_timer(self.current_time)
                    v.beacon_miss_count = 0
                    if self._get_cluster_id(vid) is None:
                        self._assign_new_cluster_id(vid)
            else:
                if v.state == VehicleStateEnum.CH:
                    # Try to join a better CH
                    best_ch = None
                    best_ccf = -1.0
                    for nid in v.stable_neighbors:
                        nv = self.vehicles[nid]
                        if nv.state == VehicleStateEnum.CH and nv.ccf > best_ccf:
                            best_ccf = nv.ccf
                            best_ch = nid
                    if best_ch is not None:
                        v.update_pcl(self.current_time)
                        v.state = VehicleStateEnum.CM
                        v.ch_id = best_ch
                        v.cluster_id = self._get_cluster_id(best_ch)
                        self.ch_to_cluster.pop(vid, None)
                    else:
                        v.update_pcl(self.current_time)
                        v.state = VehicleStateEnum.UC
                        v.ch_id = None
                        v.cluster_id = None
                        self.ch_to_cluster.pop(vid, None)

    # ---------- assign CMs and SCHs ----------
    def _assign_cms_and_schs(self) -> None:
        for vid, v in self.vehicles.items():
            if v.state != VehicleStateEnum.UC:
                continue
            best_ch = None
            best_ccf = -1.0
            for nid in v.stable_neighbors:
                nv = self.vehicles[nid]
                if nv.state == VehicleStateEnum.CH and nv.ccf > best_ccf:
                    best_ccf = nv.ccf
                    best_ch = nid
            if best_ch is not None:
                v.state = VehicleStateEnum.CM
                v.ch_id = best_ch
                v.beacon_miss_count = 0

        for vid, v in self.vehicles.items():
            if v.state != VehicleStateEnum.CH:
                continue
            best_sch = None
            best_ccf = -1.0
            for oid, ov in self.vehicles.items():
                if ov.state == VehicleStateEnum.CM and ov.ch_id == vid:
                    if ov.ccf > best_ccf:
                        best_ccf = ov.ccf
                        best_sch = oid
            if best_sch is not None:
                self.vehicles[best_sch].state = VehicleStateEnum.SCH
                self.vehicles[best_sch].ch_id = vid

    # ---------- cluster merging (Algorithm 4) ----------
    def _merge_clusters(self) -> None:
        ch_list = [vid for vid, v in self.vehicles.items() if v.state == VehicleStateEnum.CH]
        ch_list.sort(key=lambda vid: self.vehicles[vid].ccf, reverse=True)
        merged = set()
        for i, ch1_id in enumerate(ch_list):
            if ch1_id in merged:
                continue
            ch1 = self.vehicles[ch1_id]
            for ch2_id in ch_list[i+1:]:
                if ch2_id in merged:
                    continue
                ch2 = self.vehicles[ch2_id]
                dist = self._euclidean_dist(ch1.x, ch1.y, ch2.x, ch2.y)
                if dist <= self.comm_range:
                    if ch1.ccf >= ch2.ccf:
                        self._merge_two_clusters(ch2_id, ch1_id)
                        merged.add(ch2_id)
                    else:
                        self._merge_two_clusters(ch1_id, ch2_id)
                        merged.add(ch1_id)
                        break

    def _merge_two_clusters(self, old_ch_id: str, new_ch_id: str) -> None:
        # old_ch may or may not exist (if it left, we handle gracefully)
        old_ch = self.vehicles.get(old_ch_id)
        if old_ch:
            old_ch.update_pcl(self.current_time)
        # Transfer cluster ID
        cid = self.ch_to_cluster.pop(old_ch_id, None)
        if cid is None:
            cid = self._assign_new_cluster_id(new_ch_id)
        else:
            self.ch_to_cluster[new_ch_id] = cid
            if new_ch_id in self.vehicles:
                self.vehicles[new_ch_id].cluster_id = cid
        # Demote old CH to CM if it still exists
        if old_ch:
            old_ch.state = VehicleStateEnum.CM
            old_ch.ch_id = new_ch_id
            old_ch.cluster_id = cid
            old_ch.beacon_miss_count = 0
        # All members of old cluster follow new CH
        for vid, v in list(self.vehicles.items()):
            if v.ch_id == old_ch_id and v.state in (VehicleStateEnum.CM, VehicleStateEnum.SCH):
                v.ch_id = new_ch_id
                if v.state == VehicleStateEnum.SCH:
                    v.state = VehicleStateEnum.CM

    # ---------- maintenance ----------
    def _maintain_clusters(self) -> None:
        # Dissolve CHs with fewer than 3 vehicles (CH + CMs + SCHs)
        for vid, v in list(self.vehicles.items()):
            if v.state != VehicleStateEnum.CH:
                continue
            member_count = 0
            for oid, ov in self.vehicles.items():
                if ov.ch_id == vid and ov.state in (VehicleStateEnum.CM, VehicleStateEnum.SCH):
                    member_count += 1
            # cluster size = 1 (CH) + member_count
            if member_count < 2:  # i.e., total < 3
                self._dissolve_cluster(vid)

        # Handle beacon misses for CMs and SCHs
        for vid, v in list(self.vehicles.items()):
            if v.state not in (VehicleStateEnum.CM, VehicleStateEnum.SCH):
                continue
            if v.ch_id is None:
                v.state = VehicleStateEnum.UC
                continue

            ch = self.vehicles.get(v.ch_id)
            if ch is None or ch.state != VehicleStateEnum.CH:
                # CH is gone – try to promote SCH
                sch = self._find_sch(v.ch_id)
                if sch is not None:
                    self._promote_sch(v.ch_id, sch)
                    v.ch_id = sch
                    v.beacon_miss_count = 0
                else:
                    v.state = VehicleStateEnum.UC
                    v.ch_id = None
                continue

            dist = self._euclidean_dist(v.x, v.y, ch.x, ch.y)
            if dist > self.comm_range:
                v.beacon_miss_count += 1
                if v.beacon_miss_count >= 2:
                    sch = self._find_sch(v.ch_id)
                    if sch is not None:
                        self._promote_sch(v.ch_id, sch)
                        v.ch_id = sch
                        v.beacon_miss_count = 0
                    else:
                        v.state = VehicleStateEnum.UC
                        v.ch_id = None
            else:
                v.beacon_miss_count = 0

        # Final check: dissolve any CH that still has no members
        for vid, v in list(self.vehicles.items()):
            if v.state != VehicleStateEnum.CH:
                continue
            has_member = False
            for oid, ov in self.vehicles.items():
                if ov.ch_id == vid and ov.state in (VehicleStateEnum.CM, VehicleStateEnum.SCH):
                    has_member = True
                    break
            if not has_member:
                self._dissolve_cluster(vid)

    # ---------- helper maintenance methods ----------
    def _find_sch(self, ch_id: str) -> Optional[str]:
        for vid, v in self.vehicles.items():
            if v.state == VehicleStateEnum.SCH and v.ch_id == ch_id:
                return vid
        return None

    def _promote_sch(self, old_ch_id: str, new_ch_id: str) -> None:
        """
        Promote a SCH to CH. Handles the case where old CH may have already left.
        """
        # Transfer cluster ID (or create new)
        cid = self.ch_to_cluster.pop(old_ch_id, None)
        if cid is None:
            cid = self._assign_new_cluster_id(new_ch_id)
        else:
            self.ch_to_cluster[new_ch_id] = cid
            if new_ch_id in self.vehicles:
                self.vehicles[new_ch_id].cluster_id = cid

        # If old CH still exists, demote it and update PCL
        old_ch = self.vehicles.get(old_ch_id)
        if old_ch:
            old_ch.update_pcl(self.current_time)
            old_ch.state = VehicleStateEnum.UC
            old_ch.ch_id = None
            old_ch.cluster_id = None

        # Promote new CH
        new_ch = self.vehicles.get(new_ch_id)
        if new_ch:
            new_ch.state = VehicleStateEnum.CH
            new_ch.ch_id = new_ch_id
            new_ch.cluster_id = cid
            new_ch.reset_ch_timer(self.current_time)
            new_ch.beacon_miss_count = 0

        # Update members of the old cluster to follow new CH
        for vid, v in list(self.vehicles.items()):
            if v.ch_id == old_ch_id and v.state in (VehicleStateEnum.CM, VehicleStateEnum.SCH):
                v.ch_id = new_ch_id
                if v.state == VehicleStateEnum.SCH:
                    v.state = VehicleStateEnum.CM

    def _dissolve_cluster(self, ch_id: str) -> None:
        """
        Dissolve a cluster. The CH may or may not still be present.
        """
        # Remove CH from mapping
        self.ch_to_cluster.pop(ch_id, None)
        # If CH still exists, demote it
        ch = self.vehicles.get(ch_id)
        if ch:
            ch.update_pcl(self.current_time)
            ch.state = VehicleStateEnum.UC
            ch.ch_id = None
            ch.cluster_id = None
        # All members become UC
        for vid, v in list(self.vehicles.items()):
            if v.ch_id == ch_id:
                v.state = VehicleStateEnum.UC
                v.ch_id = None
                v.cluster_id = None
                v.beacon_miss_count = 0

    # ---------- static helpers ----------
    @staticmethod
    def _same_direction(dir_a: float, dir_b: float) -> bool:
        diff = abs(dir_a - dir_b) % 360
        if diff > 180:
            diff = 360 - diff
        return diff < 90.0

    @staticmethod
    def _euclidean_dist(x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)