"""
Produces the SFR decomposition table: separates
packet losses (dropped) from late deliveries (deadline violations), and
reports the on-time reception rate.

Usage:
    python sfr_decomposition.py --scenario emergency   --density 100
    python sfr_decomposition.py --scenario emergency   --density 240
    python sfr_decomposition.py --scenario doubleinter --density 100
    python sfr_decomposition.py --scenario doubleinter --density 240
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / 'logs' / 'aggregate_csvs'


def get_paths(scen, dens, dep):
    b = DATA_DIR
    if scen == 'emergency':
        return {
            'ECDC':       b / f'ecdc_emergency{dens}_5rsu_depart{dep}.csv',
            'MCSC':       b / f'mcsc_emergency{dens}_5rsu_depart{dep}.csv',
            'IMICLiVAN':  b / f'imiclivan_emergency{dens}_5rsu_depart{dep}.csv',
            'Zone-based': b / f'zone_emergency{dens}_5rsu_depart{dep}.csv',
        }
    else:
        return {
            'ECDC':       b / f'ecdc_doubleinter{dens}_2rsu_depart{dep}.csv',
            'MCSC':       b / f'mcsc_doubleinter{dens}_2rsu_depart{dep}.csv',
            'IMICLiVAN':  b / f'imiclivan_doubleinter{dens}_2rsu_depart{dep}.csv',
            'Zone-based': b / f'zone_doubleinter{dens}_2rsu_depart{dep}.csv',
        }


def decompose_one_row(row):
    """Return a dict of the SFR-decomposition quantities for one rep."""
    n_loss  = float(row.get('n_loss', 0.0) or 0.0)
    n_late  = float(row.get('n_late', 0.0) or 0.0)
    sfr     = float(row.get('sfr', 0.0) or 0.0)
    transit = float(row.get('amb_transit_time', 0.0) or 0.0)

    # Ambulance sends exactly 1 packet/second throughout its transit window.
    # Total failures = (n_loss + n_late) and sfr = failures / N_sent.
    # => N_sent = (n_loss + n_late) / sfr, falling back to transit time
    #    whenever n_loss + n_late == 0 (perfect service, sfr == 0).
    if sfr > 1e-9 and (n_loss + n_late) > 0:
        n_sent = (n_loss + n_late) / sfr
    else:
        n_sent = transit

    n_recv    = n_sent - n_loss
    n_on_time = n_recv - n_late
    return pd.Series(dict(
        decomp_n_sent=n_sent,
        decomp_n_recv=n_recv,
        decomp_n_on_time=n_on_time,
    ))


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else 0.0, 0.0, 0.0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    tc = stats.t.ppf(0.975, df=len(x) - 1)
    return m, m - tc * se, m + tc * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['emergency', 'doubleinter'], required=True)
    ap.add_argument('--density', type=int, required=True)
    ap.add_argument('--depart', type=int, default=150)
    a = ap.parse_args()

    paths = get_paths(a.scenario, a.density, a.depart)
    rows = []

    print(f"\nSFR decomposition: scenario={a.scenario}  N={a.density}  depart={a.depart}")
    print("=" * 118)
    hdr = (f"{'Algorithm':<12} "
           f"{'N_sent':>8} {'N_recv':>8} {'N_loss':>8} {'N_late':>8} {'N_on_time':>11} "
           f"{'SFR':>8} {'95% CI SFR':>22}")
    print(hdr)
    print("-" * 118)

    for alg, p in paths.items():
        if not p.exists():
            print(f"  [skip] {alg}: {p} not found")
            continue
        df = pd.read_csv(p)

        # Decompose per rep into new columns
        decomp = df.apply(decompose_one_row, axis=1, result_type='expand')
        df = pd.concat([df.reset_index(drop=True), decomp], axis=1)

        ns      = df['decomp_n_sent'].mean()
        nr      = df['decomp_n_recv'].mean()
        nl      = df['n_loss'].mean()
        nl_late = df['n_late'].mean()
        no      = df['decomp_n_on_time'].mean()
        sfr_m, sfr_lo, sfr_hi = ci95(df['sfr'].values)

        print(f"{alg:<12} "
              f"{ns:>8.1f} {nr:>8.1f} {nl:>8.1f} {nl_late:>8.1f} {no:>11.1f} "
              f"{sfr_m:>8.4f} [{sfr_lo:>7.4f}, {sfr_hi:>7.4f}]")

        rows.append(dict(
            scenario=a.scenario, density=a.density, algorithm=alg,
            n_sent_mean=ns, n_recv_mean=nr, n_loss_mean=nl,
            n_late_mean=nl_late, n_on_time_mean=no,
            sfr_mean=sfr_m, sfr_ci_lo=sfr_lo, sfr_ci_hi=sfr_hi,
            loss_fraction=nl / ns if ns > 0 else 0.0,
            late_fraction=nl_late / ns if ns > 0 else 0.0,
            on_time_fraction=no / ns if ns > 0 else 0.0,
        ))

    print("=" * 118)
    out = pd.DataFrame(rows)
    outpath = SCRIPT_DIR / f"sfr_decomposition_{a.scenario}_N{a.density}.csv"
    out.to_csv(outpath, index=False)
    print(f"Wrote {outpath}\n")


if __name__ == '__main__':
    main()