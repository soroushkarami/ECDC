import glob, pandas as pd, numpy as np

PATTERNS = {
    "mixed_N240":  "metrics/ecdc_emergency/clustering_data/emergency_delays240_seed*_depart150.csv",
    "double_N240": "metrics/ecdc_doubleinter/clustering_data/emergency_delays240_seed*_depart150.csv",
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