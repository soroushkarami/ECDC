import math
import random
import numpy as np
import os
from net_metrics import (
    _get_collision_probability,
    _get_collision_probability_broadcast,
    phy_success_prob,
    mac_delay_deterministic,
    RETRY_LIMIT,
)

BS_RADIUS_M = float(os.environ.get('ECDC_BS_RADIUS', '0'))  # 0 = unlimited
V2I_SHADOWING_DB = float(os.environ.get('ECDC_V2I_SHADOW_DB', '0.0'))
V2I_RAYLEIGH     = os.environ.get('ECDC_V2I_RAYLEIGH', '0') == '1'
# ------------------------------------------------------------
# Physical layer constants (IEEE 802.11p / 1609 compliant)
# ------------------------------------------------------------
COMM_RANGE = 250.0  # meters

# ---------- Overhead sizes (bytes) ----------
MAC_HEADER_BYTES = 26  # IEEE 802.11 MAC header
FCS_BYTES = 4  # Frame Check Sequence
SECURITY_BYTES = 64  # IEEE 1609.2 ECDSA signature + header
PLCP_PREAMBLE_US = 32.0  # microseconds
PLCP_HEADER_US = 8.0  # microseconds

# ---------- Contention Window (Bianchi model) ----------
CW_MIN = 15
SLOT_TIME_US = 13.0  # 802.11p slot time
SIFS_US = 32.0  # Correct for 802.11p (10 MHz)
DIFS_US = SIFS_US + 2 * SLOT_TIME_US  # 58 µs

# Default payload sizes
PACKET_SIZE_BYTES = 512  # data payload
CTRL_HELLO_BYTES = 64  # basic HELLO (without security)

# Path loss model (log‑distance with shadowing)
# Based on measurement campaigns for V2V at 5.9 GHz.
# Reference: Cheng et al., "Measurement and Modeling of Wireless
# Channel in Vehicle-to-Vehicle Environment," IEEE TVT, 2007.
PL0_DB_V2V = 47.0  # path loss at 1 m (5.9 GHz)
PATH_LOSS_EXP_V2V = 2.5
SHADOWING_STD_DB_V2V = 4.0

PL0_DB_V2I = 47.0
PATH_LOSS_EXP_V2I = 2.0
SHADOWING_STD_DB_V2I = 0.0

TX_POWER_DBM_V2V = 23.0
TX_POWER_DBM_V2I = 23.0

# Receiver sensitivity for 6 Mbps (QPSK Rate 1/2) – typical value.
RX_SENSITIVITY_DBM = -90.0

WSMP_HEADER_BYTES = 12

# Energy model
P_TX_MW = 80.0
P_RX_MW = 50.0
P_IDLE_MW = 30.0

_rng_chan = random.Random(0)
_rng_np = np.random.RandomState(0)


def set_channel_seed(seed):
    _rng_chan.seed(seed)
    _rng_np.seed(seed)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def path_loss(distance_m, link_type='v2v'):
    if distance_m < 1.0:
        distance_m = 1.0
    if link_type == 'v2v':
        pl = PL0_DB_V2V + 10 * PATH_LOSS_EXP_V2V * math.log10(distance_m)
        shadow = _rng_np.normal(0, SHADOWING_STD_DB_V2V)
    else:
        pl = PL0_DB_V2I + 10 * PATH_LOSS_EXP_V2I * math.log10(distance_m)
        shadow = _rng_np.normal(0, V2I_SHADOWING_DB)

    if link_type == 'v2v':
        rayleigh = _rng_np.rayleigh(scale=1.0)
        fading_db = -20 * math.log10(rayleigh) if rayleigh > 0 else 0.0
    elif V2I_RAYLEIGH:
        rayleigh = _rng_np.rayleigh(scale=1.0)
        fading_db = -20 * math.log10(rayleigh) if rayleigh > 0 else 0.0
    else:
        fading_db = 0.0

    return pl + shadow + fading_db


def is_link_successful(distance_m, link_type='v2v'):
    """True if received power >= sensitivity."""
    if link_type == 'v2v':
        tx_power = TX_POWER_DBM_V2V
    else:
        tx_power = TX_POWER_DBM_V2I
    loss_db = path_loss(distance_m, link_type)
    rx_power_dbm = tx_power - loss_db
    return rx_power_dbm >= RX_SENSITIVITY_DBM


def packet_success_prob(distance_m, num_contenders, link_type='v2v',
                        load_factor=1.0, mode=None):
    # Data packets are unicast with retries; only control uses broadcast.
    if mode is None:
        mode = 'unicast_retry'
    tx_power = TX_POWER_DBM_V2V if link_type == 'v2v' else TX_POWER_DBM_V2I
    loss_db = path_loss(distance_m, link_type)
    rx_power_dbm = tx_power - loss_db
    if rx_power_dbm < RX_SENSITIVITY_DBM:
        return 0.0
    snr = rx_power_dbm - (-95.0)
    if link_type == 'v2i':
        snr += 3.0
    phy_success = phy_success_prob(snr)
    if mode == 'broadcast':
        p = _get_collision_probability_broadcast(num_contenders, load_factor)
        mac_success = 1.0 - p
    else:
        p = _get_collision_probability(num_contenders, load_factor)
        mac_success = 1.0 - p ** RETRY_LIMIT
    return phy_success * mac_success


# ------------------------------------------------------------
# Main metrics collector class
# ------------------------------------------------------------
class NetworkMetrics:
    def __init__(self, sim_time_total):
        self.sim_time = sim_time_total
        self.packets_sent = 0
        self.packets_received = 0
        self.payload_bytes_received = 0
        self.ctrl_bytes_sent = 0
        self.ctrl_packets_sent = 0
        self.delays = []
        self.energy_tx_sec = 0.0
        self.energy_rx_sec = 0.0
        self.energy_idle_sec = 0.0
        self.total_data_hops = 0

    def set_sim_time(self, sim_time):
        self.sim_time = sim_time

    def simulate_application_report(self, src_id, ch_id, bs_position,
                                    vehicles_positions, ch_positions,
                                    num_contenders_v2v=None, num_contenders_v2i=None,
                                    num_contenders=None, load_factor=1.0,
                                    is_emergency=False):
        """
        Simulate a data packet from src to its CH and then to BS.

        Args:
            src_id: Source vehicle ID
            ch_id: Cluster Head ID
            bs_position: (x, y) of Base Station
            vehicles_positions: dict {veh_id: (x, y)}
            ch_positions: dict {veh_id: (x, y)} (can be same as vehicles_positions)
            num_contenders_v2v: Number of contenders for V2V hop (receiver = CH)
            num_contenders_v2i: Number of contenders for V2I hop (receiver = BS)
            num_contenders: (backward compatibility) single value used for both hops

        Returns:
            (success, total_delay, hops)
        """
        if BS_RADIUS_M > 0:
            if src_id == ch_id:
                d = math.hypot(ch_positions[ch_id][0] - bs_position[0],
                               ch_positions[ch_id][1] - bs_position[1])
                if d > BS_RADIUS_M:
                    return False, 0.0, 0
            else:
                # two-hop: only the second hop (CH -> BS) matters here
                d = math.hypot(ch_positions[ch_id][0] - bs_position[0],
                               ch_positions[ch_id][1] - bs_position[1])
                if d > BS_RADIUS_M:
                    return False, 0.0, 0


        # Backward compatibility: if old-style call with single num_contenders
        if num_contenders is not None:
            num_contenders_v2v = num_contenders
            num_contenders_v2i = num_contenders

        # If still None, default to 1 (no contention)
        if num_contenders_v2v is None:
            num_contenders_v2v = 1
        if num_contenders_v2i is None:
            num_contenders_v2i = 1

        delay = 0.0

        if src_id == ch_id:
            # 1 hop: CH sends directly to BS (V2I only)
            ch_pos = ch_positions[ch_id]
            dist = math.hypot(ch_pos[0] - bs_position[0],
                              ch_pos[1] - bs_position[1])
            success, hop_delay = self._simulate_hop(dist, num_contenders_v2i, link_type='v2i', load_factor=load_factor)
            if not success:
                return False, 0.0, 0
            delay += hop_delay
            if not is_emergency:
                self.payload_bytes_received += PACKET_SIZE_BYTES
                self.packets_received += 1
            return True, delay, 1  # 1 hop

        else:
            # 2 hops: src -> CH (V2V), then CH -> BS (V2I)
            src_pos = vehicles_positions[src_id]
            ch_pos = ch_positions[ch_id]

            # Hop 1: V2V (src to CH) - receiver is CH
            dist_v2v = math.hypot(src_pos[0] - ch_pos[0],
                                  src_pos[1] - ch_pos[1])
            success_v2v, delay_v2v = self._simulate_hop(dist_v2v, num_contenders_v2v, link_type='v2v',
                                                        load_factor=load_factor)
            if not success_v2v:
                return False, 0.0, 0
            delay += delay_v2v

            # Hop 2: V2I (CH to BS) - receiver is BS
            dist_v2i = math.hypot(ch_pos[0] - bs_position[0],
                                  ch_pos[1] - bs_position[1])
            success_v2i, delay_v2i = self._simulate_hop(dist_v2i, num_contenders_v2i, link_type='v2i',
                                                        load_factor=load_factor)
            if not success_v2i:
                return False, 0.0, 0
            delay += delay_v2i

            if not is_emergency:
                self.payload_bytes_received += PACKET_SIZE_BYTES
                self.packets_received += 1
            return True, delay, 2  # 2 hops

    def _simulate_hop(self, distance, num_contenders, link_type='v2v', load_factor=1.0):
        prop_delay = distance / 3e8
        data_rate = 6_000_000
        total_bytes = (PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES +
                       SECURITY_BYTES + WSMP_HEADER_BYTES)
        tx_delay = (PLCP_PREAMBLE_US + PLCP_HEADER_US) * 1e-6 + (total_bytes * 8) / data_rate

        mode = 'unicast_retry'
        mac_delay = mac_delay_deterministic(num_contenders,
                                            load_factor=load_factor,
                                            mode=mode)
        proc_delay = 0.001
        total_delay = prop_delay + tx_delay + mac_delay + proc_delay

        success_prob = packet_success_prob(distance, num_contenders, link_type,
                                           load_factor=load_factor,
                                           mode=mode)
        success = _rng_chan.random() < success_prob
        return success, total_delay

    def record_control_overhead(self, ctrl_bytes, num_packets=1):
        self.ctrl_bytes_sent += ctrl_bytes
        self.ctrl_packets_sent += num_packets

    def update_energy(self, tx_seconds, rx_seconds, idle_seconds):
        self.energy_tx_sec += tx_seconds
        self.energy_rx_sec += rx_seconds
        self.energy_idle_sec += idle_seconds

    def record_data_hops(self, hops):
        self.total_data_hops += hops

    def record_delay(self, delay_seconds):
        self.delays.append(delay_seconds)

    def get_pdr(self):
        return self.packets_received / self.packets_sent if self.packets_sent > 0 else 0.0

    def get_plr(self):
        return 1.0 - self.get_pdr()

    def get_throughput_bps(self):
        if self.sim_time == 0:
            return 0.0
        return (self.payload_bytes_received * 8) / self.sim_time

    def get_avg_delay(self):
        return sum(self.delays) / len(self.delays) if self.delays else 0.0

    def get_jitter(self):
        if len(self.delays) < 2:
            return 0.0
        diffs = [abs(self.delays[i] - self.delays[i - 1]) for i in range(1, len(self.delays))]
        return sum(diffs) / len(diffs)

    def get_overhead_bps(self):
        return self.ctrl_bytes_sent / self.sim_time if self.sim_time > 0 else 0.0

    def get_energy_efficiency(self):
        payload_bits = self.payload_bytes_received * 8
        total_packets = self.packets_sent + self.ctrl_packets_sent
        plcp_overhead_time = total_packets * (PLCP_PREAMBLE_US + PLCP_HEADER_US) * 1e-6
        total_energy = (P_TX_MW * (self.energy_tx_sec + plcp_overhead_time) +
                        P_RX_MW * self.energy_rx_sec +
                        P_IDLE_MW * self.energy_idle_sec) * 1e-3
        if total_energy == 0:
            return 0.0
        return payload_bits / total_energy

    def get_summary(self):
        return {
            'PDR': self.get_pdr(),
            'PLR': self.get_plr(),
            'Throughput (bps)': self.get_throughput_bps(),
            'Avg Delay (s)': self.get_avg_delay(),
            'Jitter (s)': self.get_jitter(),
            'Overhead (Bps)': self.get_overhead_bps(),
            'Energy Efficiency (bits/J)': self.get_energy_efficiency()
        }