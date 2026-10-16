"""
Reads the aggregate per-run CSVs and reports the maximum successful
emergency-packet latency (emdd_max) per algorithm and scenario.

Reproduces the Max (ms) column
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / 'logs' / 'aggregate_csvs'

ALGS = {
    "ECDC":       ("ecdc_emergency240_5rsu_depart150.csv",
                   "ecdc_doubleinter240_2rsu_depart150.csv"),
    "MCSC":       ("mcsc_emergency240_5rsu_depart150.csv",
                   "mcsc_doubleinter240_2rsu_depart150.csv"),
    "IMICLiVAN":  ("imiclivan_emergency240_5rsu_depart150.csv",
                   "imiclivan_doubleinter240_2rsu_depart150.csv"),
    "Zone-based": ("zone_emergency240_5rsu_depart150.csv",
                   "zone_doubleinter240_2rsu_depart150.csv"),
}

for alg, (m_file, d_file) in ALGS.items():
    for label, path in [("mixed", DATA_DIR / m_file),
                        ("double", DATA_DIR / d_file)]:
        try:
            df = pd.read_csv(path)
            v = df["emdd_max"].mean() * 1e3
            print(f"{alg:12s} {label:6s}  max-latency = {v:7.2f} ms  (n={len(df)})")
        except FileNotFoundError:
            print(f"{alg:12s} {label:6s}  FILE NOT FOUND: {path}")