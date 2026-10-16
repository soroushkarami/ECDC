"""
NOTE: This script requires the per-vehicle network logs under
metrics/*/network_data/*.txt, which are NOT distributed with this
repository because of their size (hundreds of MB across all runs).

To run it, re-run the SUMO simulations to regenerate the
network_data logs, then place them under metrics/*/network_data/ and
run this script from the repository root.
"""
import glob, os, re, statistics
from collections import defaultdict

T_DEPART   = 150.0
HELLO_RATE = 13_600.0   # 170 B × 8 bits × 10 Hz

def active_population_per_step(path):
    pop = defaultdict(int)
    with open(path, "r", errors="ignore") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            pop[t] += 1
    return pop

def parse(path):
    fn = os.path.basename(path)
    scen = ("double" if "doubleinter" in fn or "double_inter" in fn
            else "mixed" if "emergency" in fn else "?")
    m = re.search(r"(?:emergency|doubleinter)(\d+)", fn)
    dens = int(m.group(1)) if m else None
    ms = re.search(r"seed(\d+)", fn)
    seed = int(ms.group(1)) if ms else None
    return scen, dens, seed

def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    return dict(mean=statistics.mean(vals),
                min=min(vals), max=max(vals),
                p95=s[int(0.95 * len(s)) - 1], n=len(vals))

def main():
    files = [f for f in sorted(glob.glob("metrics/*/network_data/*.txt"))
             if "_old" not in f and "/_old/" not in f]

    # Deduplicate: mobility is identical across algorithms per (scen, dens, seed)
    buckets = defaultdict(list)
    seen = defaultdict(set)
    for f in files:
        scen, dens, seed = parse(f)
        if scen == "?" or dens is None or seed is None:
            continue
        if seed in seen[(scen, dens)]:
            continue
        seen[(scen, dens)].add(seed)
        buckets[(scen, dens)].append(active_population_per_step(f))

    print("=" * 108)
    print(f"{'Scen':<7} {'N':>4} {'Seeds':>6} | "
          f"{'Full-sim mean':>14} {'Transit mean':>13} {'Max':>5} | "
          f"{'Reported OH':>12} {'Implied N':>10}")
    print("=" * 108)

    # Reported overhead values (Mbps)
    reported_oh = {("mixed", 100): 0.604,
                   ("mixed", 240): 2.065,
                   ("double", 100): 0.783,
                   ("double", 240): 2.84}

    rows = {}
    for key in [("mixed", 100), ("mixed", 240),
                ("double", 100), ("double", 240)]:
        if key not in buckets:
            continue
        full_vals, trans_vals, all_t = [], [], set()
        for pop in buckets[key]:
            full_vals.extend(pop.values())
            trans_vals.extend(n for t, n in pop.items() if t >= T_DEPART)
            all_t.update(pop.keys())
        sf = stats(full_vals)
        st = stats(trans_vals)
        oh = reported_oh[key]
        implied = oh * 1e6 / HELLO_RATE
        rows[key] = (sf, st)
        print(f"{key[0]:<7} {key[1]:>4} {len(buckets[key]):>6} | "
              f"{sf['mean']:>14.1f} {st['mean']:>13.1f} {sf['max']:>5} | "
              f"{oh:>12.3f} {implied:>10.1f}")

    print("=" * 108)
    print("\nREADY-TO-PASTE ROWS")
    for key in [("mixed", 100), ("mixed", 240),
                ("double", 100), ("double", 240)]:
        if key not in rows:
            continue
        sf, st = rows[key]
        label = "Mixed" if key[0] == "mixed" else "Double"
        print(f"{label}, $N={key[1]}$ & {sf['mean']:.1f} & {sf['min']} "
              f"& {st['mean']:.1f} & {st['max']} \\\\")

if __name__ == "__main__":
    main()