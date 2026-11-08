#!/bin/bash
# ============================================================
# nojaccard ablation only
# 4 configs x 5 seeds = 20 sims
# Same DEPART and seeds as run_batch.sh so results are comparable
# ============================================================

export PYTHONHASHSEED=0

set -u

DEPART=150
MIXED_SEED=43     # 43,44,45,46,47
DOUBLE_SEED=42    # 42,43,44,45,46
REPS=5
DENSITIES="100 240"

echo "=========================================="
echo "NOJACCARD ABLATION BATCH"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
echo "=========================================="
date

# ---------- Phase 1: back up existing nojaccard CSVs (if any) ----------
echo ""
echo "[Phase 1] Backing up existing nojaccard CSVs..."
for folder in \
    metrics/ecdc_emergency/clustering_data \
    metrics/ecdc_doubleinter/clustering_data
do
    if ls ${folder}/*nojaccard* >/dev/null 2>&1; then
        backup_dir="${folder}_nojaccard_prev"
        rm -rf "$backup_dir"
        mkdir -p "$backup_dir"
        mv ${folder}/*nojaccard* "$backup_dir"/ 2>/dev/null || true
        echo "  backed up nojaccard files from $folder to $backup_dir"
    else
        echo "  no existing nojaccard files in $folder"
    fi
done

# ---------- Phase 2: nojaccard runs ----------
echo ""
echo "[Phase 2] nojaccard runs (20 sims)..."
for dens in $DENSITIES; do
    # Mixed
    echo "  RUN  ecdc_emergency.py  N=$dens  ablation=nojaccard  seed=$MIXED_SEED"
    ECDC_ABLATION=nojaccard python ecdc_emergency.py \
        --density $dens --seed $MIXED_SEED --reps $REPS --amb_depart $DEPART

    # Double
    echo "  RUN  ecdc_doubleinter.py  N=$dens  ablation=nojaccard  seed=$DOUBLE_SEED"
    ECDC_ABLATION=nojaccard python ecdc_doubleinter.py \
        --density $dens --seed $DOUBLE_SEED --reps $REPS --amb_depart $DEPART
done

echo ""
echo "=========================================="
echo "NOJACCARD BATCH COMPLETE"
date
echo "=========================================="