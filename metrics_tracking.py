"""
Packet, link, and event tracking utilities.
"""
import math

class PacketCounters:
    """Separate ordinary vs emergency accounting; track disconnection."""

    def __init__(self):
        self.ordinary_sent = 0
        self.ordinary_recv = 0
        self.ordinary_hops = 0
        self.ordinary_tx_bytes = 0

        self.emergency_sent = 0
        self.emergency_recv = 0
        self.emergency_late = 0
        self.emergency_hops = 0
        self.emergency_tx_bytes = 0

        self.disconnected_seconds = 0
        self.total_ambulance_seconds = 0

    def get_ordinary_pdr(self):
        return self.ordinary_recv / self.ordinary_sent if self.ordinary_sent else 0.0

    def get_emergency_pdr(self):
        return self.emergency_recv / self.emergency_sent if self.emergency_sent else 0.0

    def get_emergency_sfr(self):
        if not self.emergency_sent:
            return 0.0
        lost = self.emergency_sent - self.emergency_recv
        return (lost + self.emergency_late) / self.emergency_sent


class EpisodeTracker:
    """
    Generic (vehicle, head, start) episode tracker.

      close_on_range=True  -> episode ends when distance to head > comm_range
                              (used for V2V link duration).
      close_on_range=False -> range is ignored (used for CH tenure and
                              member-to-CH assignment duration).

    Episodes always end on: head identity change, head disappearance,
    validity loss, or vehicle departure.
    """

    def __init__(self, comm_range, close_on_range=True):
        self.comm_range = comm_range
        self.close_on_range = close_on_range
        self.open_episodes = {}
        self.durations = []

    def update(self, time_step, vehicle_ids, positions, head_of, valid):
        present = set(vehicle_ids)

        # Close episodes for departed vehicles at the time they left
        for vid in list(self.open_episodes.keys()):
            if vid not in present:
                ep = self.open_episodes.pop(vid)
                dur = time_step - ep['start_time']
                if dur > 0:
                    self.durations.append((vid, ep['head_id'], dur))

        # Update remaining vehicles
        for vid in present:
            new_head = head_of.get(vid)
            is_valid = valid.get(vid, False)
            old = self.open_episodes.get(vid)

            should_close = False
            if old is not None:
                if not is_valid or new_head is None:
                    should_close = True
                elif old['head_id'] != new_head:
                    should_close = True
                elif self.close_on_range:
                    p_v = positions.get(vid)
                    p_h = positions.get(old['head_id'])
                    if p_v is None or p_h is None:
                        should_close = True
                    else:
                        d2 = (p_v[0] - p_h[0]) ** 2 + (p_v[1] - p_h[1]) ** 2
                        if d2 > self.comm_range ** 2:
                            should_close = True

            if should_close and old is not None:
                dur = time_step - old['start_time']
                if dur > 0:
                    self.durations.append((vid, old['head_id'], dur))
                self.open_episodes.pop(vid, None)
                old = None

            if old is None and is_valid and new_head is not None:
                self.open_episodes[vid] = {'head_id': new_head,
                                           'start_time': time_step}

    def finalize(self, final_time):
        for vid, ep in list(self.open_episodes.items()):
            dur = final_time - ep['start_time']
            if dur > 0:
                self.durations.append((vid, ep['head_id'], dur))
        self.open_episodes.clear()

    def average(self):
        if not self.durations:
            return 0.0
        return sum(d for _, _, d in self.durations) / len(self.durations)

    def median(self):
        if not self.durations:
            return 0.0
        s = sorted(d for _, _, d in self.durations)
        return s[len(s) // 2]

    def maximum(self):
        if not self.durations:
            return 0.0
        return max(d for _, _, d in self.durations)



class UnifiedEventLog:
    """Single source for cluster births, CH transfers, dissolutions """

    def __init__(self):
        self.events = []  # list of dicts
        self._last_pos = {}  # cluster_id -> (x, y) last known position

    def record(self, time_step, event_type, cluster_id, position,
               ch_id=None, old_ch_id=None):
        # event_type ∈ {'birth', 'transfer', 'dissolution'}
        self.events.append({
            'time': time_step, 'type': event_type,
            'cluster_id': cluster_id, 'position': position,
            'ch_id': ch_id, 'old_ch_id': old_ch_id,
        })

    def note_ch_position(self, cluster_id, position):
        self._last_pos[cluster_id] = position

    def last_position(self, cluster_id, default=(0.0, 0.0)):
        return self._last_pos.get(cluster_id, default)

    def lcd_events(self, amb_positions, radius=250.0):
        seen = set()
        for ev in self.events:
            amb = amb_positions.get(ev['time'])
            if amb is None:
                continue
            if math.hypot(ev['position'][0] - amb[0],
                          ev['position'][1] - amb[1]) <= radius:
                seen.add((ev['time'], ev['position']))
        return len(seen)

    def count(self, event_type):
        return sum(1 for ev in self.events if ev['type'] == event_type)

    def lcd_events_up_to(self, amb_positions, time_step, radius=250.0):
        seen = set()
        for ev in self.events:
            if ev['time'] > time_step:
                continue
            amb = amb_positions.get(ev['time'])
            if amb is None:
                continue
            if math.hypot(ev['position'][0] - amb[0],
                          ev['position'][1] - amb[1]) <= radius:
                seen.add((ev['time'], ev['position']))
        return len(seen)