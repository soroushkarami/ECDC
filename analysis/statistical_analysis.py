import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / 'logs' / 'aggregate_csvs'

METRICS = [
    'pdr', 'throughput_bps', 'avg_delay', 'jitter',
    'overhead_bps', 'energy_bits_per_joule',
    'coverage_stability', 'link_stability_ratio', 'avg_link_duration',
    'avg_ch_tenure', 'avg_cluster_lifetime', 'total_ch_changes',
    'total_clusters_created',
    'eicd_total', 'ccr_avg', 'emdd_avg', 'sfr', 'ambulance_pdr',
    # optional (will be silently skipped if missing):
    'n_loss', 'n_late', 'n_dissolutions',
]

LABELS = {
    'pdr':'PDR (%)', 'throughput_bps':'Throughput', 'avg_delay':'Delay (ms)',
    'jitter':'Jitter (ms)', 'overhead_bps':'Overhead',
    'energy_bits_per_joule':'Energy eff.',
    'coverage_stability':'Coverage stab.', 'link_stability_ratio':'Link stab.',
    'avg_link_duration':'Avg link dur. (s)', 'avg_ch_tenure':'CH tenure (s)',
    'avg_cluster_lifetime':'Cluster lifetime (s)', 'total_ch_changes':'CH changes',
    'total_clusters_created':'Clusters created',
    'eicd_total':'LCD', 'ccr_avg':'CCR', 'emdd_avg':'EMDD (ms)',
    'sfr':'SFR', 'ambulance_pdr':'Ambulance PDR',
    'n_loss':'N_loss', 'n_late':'N_late', 'n_dissolutions':'N_dissolutions',
}

def get_path(scen, dens, dep, alg):
    b = DATA_DIR
    if scen == 'emergency':
        d = {
            'ECDC':      'ecdc_emergency{d}_5rsu_depart{p}.csv',
            'MCSC':      'mcsc_emergency{d}_5rsu_depart{p}.csv',
            'IMICLiVAN': 'imiclivan_emergency{d}_5rsu_depart{p}.csv',
            'Zone-based':'zone_emergency{d}_5rsu_depart{p}.csv',
        }
    else:
        d = {
            'ECDC':      'ecdc_doubleinter{d}_2rsu_depart{p}.csv',
            'MCSC':      'mcsc_doubleinter{d}_2rsu_depart{p}.csv',
            'IMICLiVAN': 'imiclivan_doubleinter{d}_2rsu_depart{p}.csv',
            'Zone-based':'zone_doubleinter{d}_2rsu_depart{p}.csv',
        }
    return b / d[alg].format(d=dens, p=dep)


def load(p):
    if not p.exists():
        print(f"  [WARN] not found: {p}")
        return None
    df = pd.read_csv(p)
    if 'rep' not in df.columns or df.empty:
        print(f"  [WARN] bad CSV: {p}")
        return None
    return df


def paired(e_df, b_df, metrics):
    merged = e_df.merge(b_df, on='rep', suffixes=('_e', '_b'))
    if len(merged) < 2:
        return {}
    out = {}
    for m in metrics:
        ce, cb = f'{m}_e', f'{m}_b'
        if ce not in merged.columns or cb not in merged.columns:
            continue
        e = merged[ce].to_numpy(dtype=float)
        b = merged[cb].to_numpy(dtype=float)
        mask = ~(np.isnan(e) | np.isnan(b))
        e, b = e[mask], b[mask]
        if len(e) < 2:
            continue
        diff = e - b
        n = len(diff)
        md = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(n))
        tc = float(stats.t.ppf(0.975, df=n - 1))
        try:
            _, p = stats.ttest_rel(e, b)
            p = float(p)
        except Exception:
            p = float('nan')
        out[m] = (md, md - tc * se, md + tc * se, p, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['emergency', 'doubleinter'], required=True)
    ap.add_argument('--density', type=int, required=True)
    ap.add_argument('--depart', type=int, default=150)
    a = ap.parse_args()

    print(f"\n{'='*104}")
    print(f"Paired analysis: scenario={a.scenario}, N={a.density}, depart={a.depart}")
    print(f"{'='*104}\n")

    ecdc = load(get_path(a.scenario, a.density, a.depart, 'ECDC'))
    if ecdc is None:
        sys.exit("ECDC CSV missing.")

    results = {}
    for base in ['MCSC', 'IMICLiVAN', 'Zone-based']:
        bdf = load(get_path(a.scenario, a.density, a.depart, base))
        if bdf is None:
            continue
        for m, v in paired(ecdc, bdf, METRICS).items():
            results[(m, base)] = v

    if not results:
        sys.exit("No comparisons produced.")

    print(f"{'Metric':<22} {'vs.':<12} {'mean Δ (E−B)':>16} {'95% CI':>30} {'p':>10} {'n':>4}")
    print('-' * 104)
    for (m, b), (md, cl, ch, p, n) in results.items():
        ci = f"[{cl:+.3f}, {ch:+.3f}]"
        ps = f"{p:.4f}" if not np.isnan(p) else "n/a"
        sig = ''
        if not np.isnan(p):
            if p < 0.01: sig = ' **'
            elif p < 0.05: sig = ' *'
        print(f"{LABELS.get(m, m):<22} {b:<12} {md:>+16.4f} {ci:>30} {ps:>10} {n:>4}{sig}")

    print('\n' + '=' * 104)
    print('* p<0.05, ** p<0.01. Positive Δ = ECDC higher than baseline.')
    print('CI crossing zero  =>  statistical TIE (do not claim advantage).')
    print('=' * 104 + '\n')

    out = pd.DataFrame([
        {'metric': m, 'baseline': b, 'mean_diff': md,
         'ci_lo': cl, 'ci_hi': ch, 'p_value': p, 'n_reps': n,
         'ci_crosses_zero': (cl * ch < 0)}
        for (m, b), (md, cl, ch, p, n) in results.items()
    ])
    outpath = SCRIPT_DIR / f"paired_stats_{a.scenario}_N{a.density}_depart{a.depart}.csv"
    out.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")


if __name__ == '__main__':
    main()