"""
Reconciles the LCD metric with the network-wide CH-change and dissolution
counters, using the aggregate CSVs. LCD counts two event types within
250 m of the ambulance: (a) CH transfers within a surviving cluster and
(b) cluster dissolutions. total_ch_changes counts only (a) network-wide.
n_dissolutions counts only (b) network-wide. LCD is therefore a
spatially-filtered subset of (a)+(b).

Usage:
    python lcd_reconciliation.py --scenario emergency   --density 240
    python lcd_reconciliation.py --scenario doubleinter --density 240
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['emergency', 'doubleinter'], required=True)
    ap.add_argument('--density', type=int, required=True)
    ap.add_argument('--depart', type=int, default=150)
    a = ap.parse_args()

    paths = get_paths(a.scenario, a.density, a.depart)
    rows = []

    print(f"\nLCD reconciliation: scenario={a.scenario}  N={a.density}  depart={a.depart}")
    print("=" * 118)
    hdr = (f"{'Algorithm':<12} "
           f"{'LCD':>8} {'CH_trans':>10} {'Dissol':>8} {'Net_total':>11} "
           f"{'LCD / Net_total':>17} {'Check (<=1)':>12}")
    print(hdr)
    print("-" * 118)

    for alg, p in paths.items():
        if not p.exists():
            print(f"  [skip] {alg}: {p} not found")
            continue
        df = pd.read_csv(p)
        lcd   = df['eicd_total'].mean()
        ch    = df['total_ch_changes'].mean()
        diss  = df['n_dissolutions'].mean()
        net   = ch + diss
        ratio = lcd / net if net > 0 else float('nan')
        ok    = "OK" if ratio <= 1.0 + 1e-6 else "FAIL (>1)"

        print(f"{alg:<12} "
              f"{lcd:>8.0f} {ch:>10.0f} {diss:>8.0f} {net:>11.0f} "
              f"{ratio:>17.3f} {ok:>12}")

        rows.append(dict(
            scenario=a.scenario, density=a.density, algorithm=alg,
            LCD=lcd, CH_transfers=ch, dissolutions=diss,
            network_total=net, LCD_over_net=ratio,
            check=('pass' if ratio <= 1.0 + 1e-6 else 'fail'),
        ))

    print("=" * 118)
    print("Interpretation:")
    print("  LCD          = CH transfers + dissolutions, within 250 m of the ambulance.")
    print("  CH_trans     = leadership transfers within SURVIVING clusters, network-wide.")
    print("  Dissol       = cluster dissolutions (anywhere), network-wide.")
    print("  Net_total    = CH_trans + Dissol (network-wide event total).")
    print("  Ratio <= 1 confirms LCD is a spatial subset of network-wide events.\n")

    out = pd.DataFrame(rows)
    outpath = SCRIPT_DIR / f"lcd_reconciliation_{a.scenario}_N{a.density}.csv"
    out.to_csv(outpath, index=False)
    print(f"Wrote {outpath}\n")


if __name__ == '__main__':
    main()