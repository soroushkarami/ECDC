"""
Standalone analytical sensitivity of the modelled PHY layer to impairments.

Reimplements the exact V2V formulas from net_metrics.py to compute the
expected PHY-layer success probability under three channel conditions:

  1. 'free'   — free-space path loss only (no shadowing, no fading)
  2. 'shadow' — free-space + log-normal shadowing (X_sigma ~ N(0, 4^2) dB)
  3. 'full'   — free-space + shadowing + Rayleigh fading (paper's full model)

Does NOT touch net_metrics.py and does NOT require SUMO.
Usage: python impairment_sensitivity.py
"""
import math
import numpy as np

# --- Parameters copied verbatim from net_metrics.py ---
PL0_DB_V2V           = 47.0
PATH_LOSS_EXP_V2V    = 2.5
SHADOWING_STD_DB_V2V = 4.0
TX_POWER_DBM_V2V     = 23.0
RX_SENSITIVITY_DBM   = -90.0
NOISE_FLOOR_DBM      = -95.0

# PER logistic curve (identical to net_metrics.phy_success_prob)
def phy_success_prob(snr_db):
    a, b = 0.8, 6.0
    per = 1.0 / (1.0 + math.exp(a * (snr_db - b)))
    return 1.0 - per

def _success_from_loss(loss_db):
    rx_dbm = TX_POWER_DBM_V2V - loss_db
    if rx_dbm < RX_SENSITIVITY_DBM:
        return 0.0
    snr = rx_dbm - NOISE_FLOOR_DBM
    return phy_success_prob(snr)

def expected_phy_success(distance_m, mode, n_mc=1_000_000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if distance_m < 1.0:
        distance_m = 1.0
    pl_mean = PL0_DB_V2V + 10.0 * PATH_LOSS_EXP_V2V * math.log10(distance_m)
    # Vectorised MC: draw all samples at once
    pl = np.full(n_mc, pl_mean)
    if mode in ('shadow', 'full'):
        pl = pl + rng.normal(0.0, SHADOWING_STD_DB_V2V, size=n_mc)
    if mode == 'full':
        h = rng.rayleigh(scale=1.0, size=n_mc)
        # Guard against log10(0)
        h = np.clip(h, 1e-9, None)
        pl = pl + (-20.0 * np.log10(h))
    rx = TX_POWER_DBM_V2V - pl
    snr = rx - NOISE_FLOOR_DBM
    alive = rx >= RX_SENSITIVITY_DBM
    per = 1.0 / (1.0 + np.exp(0.8 * (snr - 6.0)))
    succ = (1.0 - per) * alive
    return float(np.mean(succ))

def main():
    rng = np.random.default_rng(42)
    print(f"{'Distance (m)':>12} {'Free-space':>12} {'+Shadow':>12} {'+Rayleigh':>12}")
    print("-" * 52)
    for d in [25, 50, 100, 150, 200, 250]:
        p_free   = expected_phy_success(d, 'free',   rng=rng)
        p_shadow = expected_phy_success(d, 'shadow', rng=rng)
        p_full   = expected_phy_success(d, 'full',   rng=rng)
        print(f"{d:>12} {p_free:>12.4f} {p_shadow:>12.4f} {p_full:>12.4f}")

if __name__ == "__main__":
    main()