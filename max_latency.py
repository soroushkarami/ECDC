import pandas as pd, glob

ALGS = {
    "ECDC":       ("ecdc_emergency",     "ecdc_emergency240_5rsu_depart150.csv",
                   "ecdc_doubleinter",   "ecdc_doubleinter240_2rsu_depart150.csv"),
    "MCSC":       ("mcsc_emergency",     "mcsc_emergency240_5rsu_depart150.csv",
                   "mcsc_doubleinter",   "mcsc_doubleinter240_2rsu_depart150.csv"),
    "IMICLiVAN":  ("imiclivan_emergency","imiclivan_emergency240_5rsu_depart150.csv",
                   "imiclivan_doubleinter","imiclivan_doubleinter240_2rsu_depart150.csv"),
    "Zone-based": ("zone_based_emergency","zone_emergency240_5rsu_depart150.csv",
                   "zone_based_doubleinter","zone_doubleinter240_2rsu_depart150.csv"),
}

for alg, (m_dir, m_file, d_dir, d_file) in ALGS.items():
    for label, path in [("mixed", f"metrics/{m_dir}/clustering_data/{m_file}"),
                        ("double", f"metrics/{d_dir}/clustering_data/{d_file}")]:
        try:
            df = pd.read_csv(path)
            v = df["emdd_max"].mean() * 1e3
            print(f"{alg:12s} {label:6s}  max-latency = {v:7.2f} ms  (n={len(df)})")
        except FileNotFoundError:
            print(f"{alg:12s} {label:6s}  FILE NOT FOUND: {path}")