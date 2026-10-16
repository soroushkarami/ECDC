"""
NOTE: This script requires the per-vehicle network logs under
metrics/*/network_data/*.txt, which are NOT distributed with this
repository because of their size (hundreds of MB across all runs).

To run it, re-run the SUMO simulations to regenerate the
network_data logs, then place them under metrics/*/network_data/ and
run this script from the repository root.
"""
import glob
import os
import re
from collections import defaultdict
import statistics


def find_ambulance_stats(path):
    """Return dict with n, mean, min, max speed for the ambulance, or None."""
    speeds = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            if parts[1] != "ambulance":
                continue
            try:
                speeds.append(float(parts[10]))
            except ValueError:
                continue
    if not speeds:
        return None
    return {
        "n": len(speeds),
        "mean": statistics.mean(speeds),
        "min": min(speeds),
        "max": max(speeds),
    }


def parse_scenario_density_alg(path):
    """Parse scenario, density, algorithm from the file path/filename."""
    fn = os.path.basename(path)

    # Scenario
    if "doubleinter" in fn or "double_inter" in fn:
        scen = "double"
    elif "emergency" in fn:
        scen = "mixed"
    else:
        scen = "?"

    # Density
    m = re.search(r"(?:emergency|doubleinter|double_inter)(\d+)", fn)
    dens = int(m.group(1)) if m else None

    # Algorithm
    alg = "?"
    for a in ["ecdc", "mcsc", "imiclivan", "zone"]:
        if fn.lower().startswith(a):
            alg = a
            break

    return scen, dens, alg


def main():
    # Skip any backup directories created by run_batch.sh
    all_files = sorted(glob.glob("metrics/*/network_data/*.txt"))
    files = [f for f in all_files if "_old" not in f and "/_old/" not in f]

    if not files:
        print("No files found under metrics/*/network_data/*.txt")
        print("Did you run this from the project root?")
        return

    print(f"Found {len(files)} network log files (excluding *_old backups)\n")

    per_file = []
    for f in files:
        stats = find_ambulance_stats(f)
        if stats is None:
            continue
        scen, dens, alg = parse_scenario_density_alg(f)
        per_file.append((scen, dens, alg, f, stats))

    if not per_file:
        print("No files contained any 'ambulance' rows.")
        return

    # ---- Per-file table ----
    print("=" * 110)
    print("PER-FILE AMBULANCE SPEED STATISTICS")
    print("=" * 110)
    print(f"{'Scen':<7} {'N':>4} {'Alg':<11} {'File':<60} "
          f"{'n':>5} {'mean':>7} {'min':>7} {'max':>7}")
    print("-" * 110)
    for scen, dens, alg, f, s in per_file:
        print(f"{scen:<7} {dens:>4} {alg:<11} "
              f"{os.path.basename(f)[:60]:<60} "
              f"{s['n']:>5} {s['mean']:>7.2f} {s['min']:>7.2f} {s['max']:>7.2f}")

    # ---- Aggregate by scenario x density ----
    agg = defaultdict(list)
    for scen, dens, alg, f, s in per_file:
        agg[(scen, dens)].append(s)

    print()
    print("=" * 110)
    print("AGGREGATED BY SCENARIO AND DENSITY (across all algs and seeds)")
    print("=" * 110)
    print(f"{'Scen':<7} {'N':>4} {'Files':>6} {'n_total':>8} "
          f"{'mean':>7} {'min':>7} {'max':>7}")
    print("-" * 110)
    rows_for_table = {}
    for (scen, dens), slist in sorted(agg.items()):
        total_n = sum(s["n"] for s in slist)
        weighted_mean = (sum(s["mean"] * s["n"] for s in slist) / total_n
                         if total_n > 0 else 0.0)
        mn = min(s["min"] for s in slist)
        mx = max(s["max"] for s in slist)
        print(f"{scen:<7} {dens:>4} {len(slist):>6} {total_n:>8} "
              f"{weighted_mean:>7.2f} {mn:>7.2f} {mx:>7.2f}")
        rows_for_table[(scen, dens)] = (weighted_mean, mn, mx)

    # ---- Ready-to-paste table for the LaTeX ----
    print()
    print("=" * 110)
    print("READY-TO-PASTE FOR tab:amb_speed")
    print("=" * 110)
    order = [("mixed", 100), ("mixed", 240), ("double", 100), ("double", 240)]
    for key in order:
        if key not in rows_for_table:
            continue
        scen, dens = key
        mean, mn, mx = rows_for_table[key]
        label = ("Mixed" if scen == "mixed" else "Double")
        print(f"{label}, $N={dens}$ & {mean:.2f} & {mn:.2f} & {mx:.2f} \\\\")


if __name__ == "__main__":
    main()