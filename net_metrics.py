import math
import random
import numpy as np

# ------------------------------------------------------------
# Physical layer constants (IEEE 802.11p / 1609 compliant)
# ------------------------------------------------------------
COMM_RANGE = 250.0                # meters

# ---------- Overhead sizes (bytes) ----------
MAC_HEADER_BYTES = 26             # IEEE 802.11 MAC header
FCS_BYTES = 4                     # Frame Check Sequence
SECURITY_BYTES = 64               # IEEE 1609.2 ECDSA signature + header
PLCP_PREAMBLE_US = 32.0           # microseconds
PLCP_HEADER_US = 8.0             # microseconds

# ---------- Contention Window (Bianchi model) ----------
CW_MIN = 15
SLOT_TIME_US = 13.0               # 802.11p slot time
SIFS_US = 32.0                    # Correct for 802.11p (10 MHz)
DIFS_US = SIFS_US + 2 * SLOT_TIME_US   # 58 µs

# Default payload sizes
PACKET_SIZE_BYTES = 512           # data payload
CTRL_HELLO_BYTES = 64             # basic HELLO (without security)

# Path loss model (log‑distance with shadowing)
# Based on measurement campaigns for V2V at 5.9 GHz.
# Reference: Cheng et al., "Measurement and Modeling of Wireless
# Channel in Vehicle-to-Vehicle Environment," IEEE TVT, 2007.
PL0_DB_V2V = 47.0                 # path loss at 1 m (5.9 GHz)
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

# ------------------------------------------------------------
# PHY layer success probability (PER-based)
# ------------------------------------------------------------
def phy_success_prob(snr_db):
    """
    Compute PHY layer packet success probability using a logistic(sigmoid) PER model.
    The curve is calibrated to typical IEEE 802.11p PER vs. SNR for QPSK Rate 1/2:
        PER = 0.1 at SNR = 9 dB (higher snr leads to lower per)
        PER = 0.5 at SNR = 6 dB
        PER = 0.9 at SNR = 3 dB
    This yields a smooth transition from low to high PER.
    """
    # Logistic parameters: adjust a and b to fit the above points.
    # Let b = 6.0 (SNR for 50% PER), a = 0.8 controls the slope.
    a = 0.8
    b = 6.0
    # PER = 1 / (1 + exp(a * (SNR - b)))  -- this gives PER=0.5 at SNR=b.
    # Using a=0.8 gives PER(9) ≈ 0.09, PER(3) ≈ 0.92, 
    per = 1.0 / (1.0 + math.exp(a * (snr_db - b)))
    return 1.0 - per

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def path_loss(distance_m, link_type='v2v'):
    """Return path loss in dB with shadowing and Rayleigh multipath fading."""
    if distance_m < 1.0:
        distance_m = 1.0
    if link_type == 'v2v':
        pl = PL0_DB_V2V + 10 * PATH_LOSS_EXP_V2V * math.log10(distance_m)
        shadow = np.random.normal(0, SHADOWING_STD_DB_V2V)
    else:  # v2i
        pl = PL0_DB_V2I + 10 * PATH_LOSS_EXP_V2I * math.log10(distance_m)
        shadow = np.random.normal(0, SHADOWING_STD_DB_V2I)

    # Multipath fading (Rayleigh) for V2V – no clipping
    if link_type == 'v2v':
        rayleigh = np.random.rayleigh(scale=1.0)
        if rayleigh > 0:
            fading_db = -20 * math.log10(rayleigh)
        else:
            fading_db = 0.0
        # No clipping – allow deep fades up to 40–50 dB.
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


def packet_success_prob(distance_m, num_contenders, link_type='v2v', load_factor=1.0):
    """
    Success probability = PHY success (based on SNR) * MAC success (no collision).
    MAC success is derived from the Bianchi model's collision probability p.
    """
    # 1. Compute received power and SNR
    if link_type == 'v2v':
        tx_power = TX_POWER_DBM_V2V
    else:
        tx_power = TX_POWER_DBM_V2I

    loss_db = path_loss(distance_m, link_type)
    
    rx_power_dbm = tx_power - loss_db

    if rx_power_dbm < RX_SENSITIVITY_DBM:
        return 0.0      # Packet dropped immediately (too weak)

    # 2. Determine SNR (Signal-to-Noise Ratio)
    noise_floor = -95.0     # Background noise power
    snr = rx_power_dbm - noise_floor

    # For V2I we add a small margin (better channel)
    if link_type == 'v2i':
        snr += 3.0

    # 3. "PHY success probability" from PER curve (calibrated for 6 Mbps)
    phy_success = phy_success_prob(snr)

    # 4. "MAC Success Probability" (Bianchi-derived collision probability)
    p = _get_collision_probability(num_contenders, load_factor)
    mac_success = (1.0 - p) if num_contenders > 1 else 1.0

    # Final success = PHY success * MAC success
    return phy_success * mac_success


def _get_collision_probability(num_contenders, load_factor=1.0):
    """
    Extract collision probability p from Bianchi model.
    load_factor scales the effective number of contenders to account for
    non‑saturated traffic (0 < load_factor ≤ 1).
    """
    if num_contenders <= 1:
        return 0.0

    # Scale contenders by the offered load factor (at least 1)
    effective_n = max(1, int(num_contenders * load_factor))
    if effective_n <= 1:
        return 0.0

    CW_min = 15
    m = 7  # max backoff stage

    tau = 1.0 / (CW_min / 2.0 + 1)
    for _ in range(10):
        p = 1.0 - (1.0 - tau) ** (effective_n - 1)
        numerator = 2.0 * (1.0 - 2.0 * p)
        denominator = (1.0 - 2.0 * p) * (CW_min + 1) + p * CW_min * (1.0 - (2.0 * p) ** m)
        if denominator == 0:
            break
        tau_new = numerator / denominator
        tau_new = max(0.01, min(0.9, tau_new))
        if abs(tau_new - tau) < 1e-6:
            tau = tau_new
            break
        tau = tau_new
    return p


def mac_delay_deterministic(num_contenders):    # Time spent waiting (backoff) --> Bianchi model
    if num_contenders <= 1:
        return DIFS_US * 1e-6

    CW_min = 15
    CW_max = 1023
    m = 7  # max backoff stage

    # Fixed-point Bianchi model to find transmission probability tau
    tau = 1.0 / (CW_min / 2.0 + 1)  # initial guess
    for _ in range(10):
        p = 1.0 - (1.0 - tau) ** (num_contenders - 1)  # collision prob
        numerator = 2.0 * (1.0 - 2.0 * p)
        denominator = (1.0 - 2.0 * p) * (CW_min + 1) + p * CW_min * (1.0 - (2.0 * p) ** m)
        if denominator == 0:
            break
        tau = numerator / denominator
        tau = max(0.01, min(0.9, tau))  # clamp for stability

    # Average backoff slots (standard Bianchi result)
    avg_backoff_slots = (CW_min / 2.0) * ((1.0 - (2.0 * p) ** (m + 1)) / (1.0 - 2.0 * p)) if p < 0.5 else CW_max / 2.0
    avg_backoff_time = avg_backoff_slots * SLOT_TIME_US * 1e-6
    return (DIFS_US * 1e-6) + avg_backoff_time


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
                                    num_contenders=None, load_factor=1.0):
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
            success_v2v, delay_v2v = self._simulate_hop(dist_v2v, num_contenders_v2v, link_type='v2v', load_factor=load_factor)
            if not success_v2v:
                return False, 0.0, 0
            delay += delay_v2v

            # Hop 2: V2I (CH to BS) - receiver is BS
            dist_v2i = math.hypot(ch_pos[0] - bs_position[0],
                                  ch_pos[1] - bs_position[1])
            success_v2i, delay_v2i = self._simulate_hop(dist_v2i, num_contenders_v2i, link_type='v2i', load_factor=load_factor)
            if not success_v2i:
                return False, 0.0, 0
            delay += delay_v2i

            self.payload_bytes_received += PACKET_SIZE_BYTES
            self.packets_received += 1
            return True, delay, 2  # 2 hops

    def _simulate_hop(self, distance, num_contenders, link_type='v2v', load_factor=1.0):
        """Simulate a single hop (V2V or V2I) without Doppler."""
        prop_delay = distance / 3e8

        # ---- FIXED DATA RATE (6 Mbps QPSK) ----
        # Fixed 6 Mbps data rate (matches PER curve calibration).
        data_rate = 6_000_000  # 6 Mbps

        # Total packet size with MAC and security overhead
        total_bytes = (PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES +
                       SECURITY_BYTES + WSMP_HEADER_BYTES)
        # Transmission delay: PLCP preamble/header + data
        tx_delay = (PLCP_PREAMBLE_US + PLCP_HEADER_US) * 1e-6 + (total_bytes * 8) / data_rate

        # MAC access delay (Bianchi)
        mac_delay = mac_delay_deterministic(num_contenders)

        proc_delay = 0.001  # 1 ms
        total_delay = prop_delay + tx_delay + mac_delay + proc_delay

        # Success probability
        success_prob = packet_success_prob(distance, num_contenders, link_type, load_factor=load_factor)
        success = random.random() < success_prob
        # If success is True: Packet delivered.
        # If success is False: Packet lost (dropped).

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
        diffs = [abs(self.delays[i] - self.delays[i-1]) for i in range(1, len(self.delays))]
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