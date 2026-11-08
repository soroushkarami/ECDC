#!/bin/bash

export PYTHONHASHSEED=0

set -u  # don't error on unbound (some tools complain)

DEPART=150
MIXED_SEED=43     # runs seeds 43,44,45,46,47
DOUBLE_SEED=42    # runs seeds 42,43,44,45,46
REPS=5
DENSITIES="100 240"

echo "=========================================="
echo "DAY 4 BATCH"
echo "=========================================="
date

# ---------- Phase 1: backup old CSVs ----------
echo ""
echo "[Phase 1] Backing up old CSVs..."
for folder in \
    metrics/ecdc_emergency/clustering_data \
    metrics/mcsc_emergency/clustering_data \
    metrics/imiclivan_emergency/clustering_data \
    metrics/zone_based_emergency/clustering_data \
    metrics/ecdc_doubleinter/clustering_data \
    metrics/mcsc_doubleinter/clustering_data \
    metrics/imiclivan_doubleinter/clustering_data \
    metrics/zone_based_doubleinter/clustering_data
do
    if [ -d "$folder" ]; then
        rm -rf "${folder}_old"
        mv "$folder" "${folder}_old"
        mkdir -p "$folder"
        echo "  backed up $folder"
    fi
done

# ---------- Phase 2: main runs (80 sims) ----------
echo ""
echo "[Phase 2] Main runs (80 sims)..."
for dens in $DENSITIES; do
    for scenario in mixed doubleinter; do
        if [ "$scenario" = "mixed" ]; then
            seed=$MIXED_SEED
        else
            seed=$DOUBLE_SEED
        fi

        # Map (algo, scenario) -> file
        case "$scenario" in
            mixed)
                ecdc_file=ecdc_emergency.py
                mcsc_file=mcsc_emergency.py
                imic_file=imic_emergency.py
                zone_file=zone_emergency.py
                ;;
            doubleinter)
                ecdc_file=ecdc_doubleinter.py
                mcsc_file=mcsc_doubleinter.py
                imic_file=imic_doubleinter.py
                zone_file=zone_doubleinter.py
                ;;
        esac

        for algo_file in $ecdc_file $mcsc_file $zone_file; do
            echo "  RUN  $algo_file  N=$dens  seed=$seed  reps=$REPS"
            python "$algo_file" --density $dens --seed $seed --reps $REPS --amb_depart $DEPART
        done

        echo "  RUN  $imic_file  N=$dens  seed=$seed  reps=$REPS  (SLOW)"
        python "$imic_file" --density $dens --seed $seed --reps $REPS --amb_depart $DEPART
    done
done

# ---------- Phase 3: ablation runs ----------
echo ""
echo "[Phase 3] Ablations at N=240 (50 sims)..."
for ab in fixedradius nohyst randrelay nogate directbs oldrch; do
    for scenario in mixed doubleinter; do
        if [ "$scenario" = "mixed" ]; then
            seed=$MIXED_SEED
            file=ecdc_emergency.py
        else
            seed=$DOUBLE_SEED
            file=ecdc_doubleinter.py
        fi

        echo "  RUN  $file  N=240  ablation=$ab  seed=$seed"
        ECDC_ABLATION=$ab python "$file" --density 240 --seed $seed --reps $REPS --amb_depart $DEPART
    done
done

# ---------- Phase 4: auto-generate plots (if plotting script exists) ----------
echo ""
echo "[Phase 4] Regenerating plots..."
if [ -f plot.py ]; then
    for scen in emergency double_inter; do
        for dens in $DENSITIES; do
            echo "  PLOT  scenario=$scen  density=$dens  depart=$DEPART"
            python plot.py --scenario $scen --density $dens --depart $DEPART --plot-type comparison || \
                echo "    (comparison plots for $scen/$dens failed or not applicable — continuing)"
            python plot.py --scenario $scen --density $dens --depart $DEPART --plot-type fairness || \
                echo "    (fairness plots for $scen/$dens failed or not applicable — continuing)"
            python plot.py --scenario $scen --density $dens --depart $DEPART --plot-type timeseries || \
                echo "    (timeseries plots for $scen/$dens failed or not applicable — continuing)"
        done
    done
else
    echo "  plot.py not found — skipping plot generation"
fi

echo ""
echo "=========================================="
echo "BATCH COMPLETE"
date
echo "=========================================="