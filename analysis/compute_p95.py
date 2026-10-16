"""
Computes the 95th-percentile end-to-end delay of successfully delivered
emergency packets. Reads the per-packet delay logs for the double-inter
and mixed scenarios.

The per-packet logs are stored under logs/mixed/ and logs/doubleinter/
as emergency_delays{N}_seed{S}_depart150.csv.
"""
import glob
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR   = SCRIPT_DIR.parent / 'logs'

PATTERNS = {
    "mixed_N240":  str(LOGS_DIR / "mixed"       / "emergency_delays240_seed*_depart150.csv"),
    "double_N240": str(LOGS_DIR / "doubleinter" / "emergency_delays240_seed*_depart150.csv"),
}

for label, pattern in PATTERNS.items():
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not any(
        k in f for k in ["_fixedradius", "_nohyst", "_nogate", "_directbs", "_randrelay"])]
    if not files:
        print(f"{label}: no files matched {pattern}")
        continue

    delays = []
    for f in files:
        df = pd.read_csv(f)
        ok = df[df["success"] == 1]["delay_s"].values
        delays.extend(ok)

    delays = np.array(delays) * 1e3  # seconds -> milliseconds
    print(f"\n{label}  (files: {len(files)}, n={len(delays)} successful packets)")
    print(f"  P50 = {np.percentile(delays, 50):.3f} ms")
    print(f"  P95 = {np.percentile(delays, 95):.3f} ms")
    print(f"  P99 = {np.percentile(delays, 99):.3f} ms")
    print(f"  Max = {delays.max():.3f} ms")