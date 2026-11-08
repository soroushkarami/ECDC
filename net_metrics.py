import math
import random
import numpy as np
from scipy.optimize import brentq

# ============================================================
# Physical layer constants (IEEE 802.11p / 1609 compliant)
# ============================================================
COMM_RANGE = 250.0

MAC_HEADER_BYTES = 26
FCS_BYTES = 4
SECURITY_BYTES = 64
PLCP_PREAMBLE_US = 32.0
PLCP_HEADER_US = 8.0

CW_MIN = 15
CW_MAX = 1023
SLOT_TIME_US = 13.0
SIFS_US = 32.0
DIFS_US = SIFS_US + 2 * SLOT_TIME_US

PACKET_SIZE_BYTES = 512
CTRL_HELLO_BYTES = 64

PL0_DB_V2V = 47.0
PATH_LOSS_EXP_V2V = 2.5
SHADOWING_STD_DB_V2V = 4.0

PL0_DB_V2I = 47.0
PATH_LOSS_EXP_V2I = 2.0
SHADOWING_STD_DB_V2I = 0.0

TX_POWER_DBM_V2V = 23.0
TX_POWER_DBM_V2I = 23.0

RX_SENSITIVITY_DBM = -90.0

WSMP_HEADER_BYTES = 12

P_TX_MW = 80.0
P_RX_MW = 50.0
P_IDLE_MW = 30.0

W_BIANCHI = CW_MIN
M_BACKOFF = 7
RETRY_LIMIT = 8

_N_MAX_CACHE = {'value': None}
BIANCHI_DIAG = {'records': []}

_rng_chan = random.Random(0)
_rng_np = np.random.RandomState(0)


def set_channel_seed(seed):
    _rng_chan.seed(seed)
    _rng_np.seed(seed)


def _bianchi_tau(p):
    num = 2.0 * (1.0 - 2.0 * p)
    den = ((1.0 - 2.0 * p) * (W_BIANCHI + 1)
           + p * W_BIANCHI * (1.0 - (2.0 * p) ** M_BACKOFF))
    return num / den


def _bianchi_F(p, n):
    tau = _bianchi_tau(p)
    return p - (1.0 - (1.0 - tau) ** (n - 1))


def _solve_bianchi_fixed_point(n):
    """Return (p, converged, residual). p is None if no root exists in (0, 0.5)."""
    if n <= 1:
        return 0.0, True, 0.0
    p_lo, p_hi = 1e-9, 0.5 - 1e-9
    try:
        F_lo = _bianchi_F(p_lo, n)
        F_hi = _bianchi_F(p_hi, n)
    except (ValueError, ZeroDivisionError):
        return None, False, float('nan')
    if F_lo * F_hi > 0:
        return None, False, float('nan')
    p_sol = brentq(lambda p: _bianchi_F(p, n), p_lo, p_hi, xtol=1e-10, maxiter=200)
    return p_sol, True, abs(_bianchi_F(p_sol, n))


def _find_n_max_bianchi():
    if _N_MAX_CACHE['value'] is not None:
        return _N_MAX_CACHE['value']
    lo, hi = 2, 200
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, conv, _ = _solve_bianchi_fixed_point(mid)
        if conv:
            lo = mid
        else:
            hi = mid - 1
    _N_MAX_CACHE['value'] = lo
    return lo


def _get_collision_probability(num_contenders, load_factor=1.0):
    """Unicast Bianchi collision probability (with backoff escalation).
       effective_n = round(num * load_factor), floored at 2, capped at n_max."""
    if num_contenders <= 1:
        return 0.0
    effective_n = max(2, int(round(num_contenders * load_factor)))
    n_max = _find_n_max_bianchi()
    capped = effective_n > n_max
    n_used = min(effective_n, n_max)
    p_sol, converged, residual = _solve_bianchi_fixed_point(n_used)
    if not converged:
        p_sol, residual = 0.5 - 1e-9, float('nan')
    BIANCHI_DIAG['records'].append(dict(
        n_raw=num_contenders, load_factor=load_factor,
        n_eff=effective_n, n_used=n_used, capped=capped,
        p=p_sol, residual=residual))
    return p_sol


def _get_collision_probability_broadcast(num_contenders, load_factor=1.0):
    """802.11p broadcast: fixed CW=CW_MIN, no escalation, single attempt."""
    if num_contenders <= 1:
        return 0.0
    effective_n = max(2, int(round(num_contenders * load_factor)))
    tau = 2.0 / (W_BIANCHI + 1)         # 0.125
    return 1.0 - (1.0 - tau) ** (effective_n - 1)


# ============================================================
# PHY layer
# ============================================================
def phy_success_prob(snr_db):
    a, b = 0.8, 6.0
    per = 1.0 / (1.0 + math.exp(a * (snr_db - b)))
    return 1.0 - per


def path_loss(distance_m, link_type='v2v'):
    if distance_m < 1.0:
        distance_m = 1.0
    if link_type == 'v2v':
        pl = PL0_DB_V2V + 10 * PATH_LOSS_EXP_V2V * math.log10(distance_m)
        shadow = _rng_np.normal(0, SHADOWING_STD_DB_V2V)
    else:
        pl = PL0_DB_V2I + 10 * PATH_LOSS_EXP_V2I * math.log10(distance_m)
        shadow = _rng_np.normal(0, SHADOWING_STD_DB_V2I)
    if link_type == 'v2v':
        rayleigh = _rng_np.rayleigh(scale=1.0)
        fading_db = -20 * math.log10(rayleigh) if rayleigh > 0 else 0.0
    else:
        fading_db = 0.0
    return pl + shadow + fading_db


def is_link_successful(distance_m, link_type='v2v'):
    tx_power = TX_POWER_DBM_V2V if link_type == 'v2v' else TX_POWER_DBM_V2I
    loss_db = path_loss(distance_m, link_type)
    return (tx_power - loss_db) >= RX_SENSITIVITY_DBM


# ============================================================
# Packet success probability
# ============================================================
def packet_success_prob(distance_m, num_contenders, link_type='v2v',
                        load_factor=1.0, mode=None):
    # Data packets are unicast with retries; only control uses broadcast.
    # The MAC layer simulates the unicast path; broadcast HELLOs are
    # accounted in the 10 Hz control overhead, not here.
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
        # up to RETRY_LIMIT attempts; packet lost only if all collide
        mac_success = 1.0 - p ** RETRY_LIMIT

    return phy_success * mac_success


def mac_delay_deterministic(num_contenders, load_factor=1.0, mode='unicast_retry'):
    """Expected MAC access delay for a delivered packet.

       Broadcast: single attempt, mean backoff = CW_MIN/2 slots.

       Unicast:   expected total backoff accumulated before success.
                  For p < 0.5, mean_slots = (W/2) * sum_{k=0}^{m} (2p)^k
                  is the expected sum of escalation-weighted backoffs over
                  all backoff stages, which bounds the mean backoff until a
                  successful transmission (success probability 1 - p^(m+1)).
                  For p >= 0.5, CW_MAX/2 is a conservative upper bound.
                  """
    if num_contenders <= 1:
        return DIFS_US * 1e-6
    if mode == 'broadcast':
        mean_slots = CW_MIN / 2.0
        return (DIFS_US * 1e-6) + mean_slots * SLOT_TIME_US * 1e-6
    p = _get_collision_probability(num_contenders, load_factor)
    if p < 0.5:
        mean_slots = ((CW_MIN / 2.0)
                      * (1.0 - (2.0 * p) ** (M_BACKOFF + 1))
                      / (1.0 - 2.0 * p))
    else:
        mean_slots = CW_MAX / 2.0
    return (DIFS_US * 1e-6) + mean_slots * SLOT_TIME_US * 1e-6


# ============================================================
# Metrics collector
# ============================================================
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
        if num_contenders is not None:
            num_contenders_v2v = num_contenders
            num_contenders_v2i = num_contenders
        if num_contenders_v2v is None:
            num_contenders_v2v = 1
        if num_contenders_v2i is None:
            num_contenders_v2i = 1

        delay = 0.0

        if src_id == ch_id:
            ch_pos = ch_positions[ch_id]
            dist = math.hypot(ch_pos[0] - bs_position[0],
                              ch_pos[1] - bs_position[1])
            success, hop_delay = self._simulate_hop(
                dist, num_contenders_v2i, link_type='v2i', load_factor=load_factor)
            if not success:
                return False, 0.0, 0
            delay += hop_delay
            if not is_emergency:
                self.payload_bytes_received += PACKET_SIZE_BYTES
                self.packets_received += 1
            return True, delay, 1

        src_pos = vehicles_positions[src_id]
        ch_pos = ch_positions[ch_id]

        dist_v2v = math.hypot(src_pos[0] - ch_pos[0], src_pos[1] - ch_pos[1])
        success_v2v, delay_v2v = self._simulate_hop(
            dist_v2v, num_contenders_v2v, link_type='v2v', load_factor=load_factor)
        if not success_v2v:
            return False, 0.0, 0
        delay += delay_v2v

        dist_v2i = math.hypot(ch_pos[0] - bs_position[0],
                              ch_pos[1] - bs_position[1])
        success_v2i, delay_v2i = self._simulate_hop(
            dist_v2i, num_contenders_v2i, link_type='v2i', load_factor=load_factor)
        if not success_v2i:
            return False, 0.0, 0
        delay += delay_v2i

        if not is_emergency:
            self.payload_bytes_received += PACKET_SIZE_BYTES
            self.packets_received += 1
        return True, delay, 2

    def _simulate_hop(self, distance, num_contenders, link_type='v2v', load_factor=1.0):
        prop_delay = distance / 3e8
        data_rate = 6_000_000
        total_bytes = (PACKET_SIZE_BYTES + MAC_HEADER_BYTES + FCS_BYTES +
                       SECURITY_BYTES + WSMP_HEADER_BYTES)
        tx_delay = (PLCP_PREAMBLE_US + PLCP_HEADER_US) * 1e-6 + (total_bytes * 8) / data_rate

        mode = 'unicast_retry'
        mac_delay = mac_delay_deterministic(num_contenders, load_factor=load_factor, mode=mode)
        proc_delay = 0.001
        total_delay = prop_delay + tx_delay + mac_delay + proc_delay

        success_prob = packet_success_prob(
            distance, num_contenders, link_type, load_factor=load_factor, mode=mode)
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
            'Energy Efficiency (bits/J)': self.get_energy_efficiency(),
        }