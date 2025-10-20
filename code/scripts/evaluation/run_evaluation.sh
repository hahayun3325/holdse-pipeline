# File: code/scripts/evaluation/run_evaluation.sh
#!/bin/bash
# Complete evaluation pipeline for GHOP model

set -e

CKPT_DIR="logs/b2e4b039a"
GHOP_DIR="$HOME/Projects/ghop/data/HOI4D_clip/Bottle_1"
OUTPUT_DIR="evaluation_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "GHOP MODEL EVALUATION PIPELINE"
echo "================================================================"
echo "Checkpoint: $CKPT_DIR"
echo "GHOP Data: $GHOP_DIR"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

mkdir -p $OUTPUT_DIR

# ================================================================
# PHASE 1: SETUP
# ================================================================
echo ""
echo "[1/4] PHASE 1: Setup - Extracting ground truth..."
echo ""

python scripts/evaluation/prepare_ghop_gt.py \
    --ghop_dir $GHOP_DIR \
    --output $OUTPUT_DIR/ghop_gt.pth

# ================================================================
# PHASE 2: QUALITATIVE EVALUATION
# ================================================================
echo ""
echo "[2/4] PHASE 2: Qualitative - Rendering visualizations..."
echo ""

python render.py \
    --load_ckpt $CKPT_DIR/checkpoints/last.ckpt \
    --config confs/ghop_production_bottle_1.yaml \
    --case ghop_bottle_1 \
    --gpu_id 0

# ================================================================
# PHASE 3: QUANTITATIVE EVALUATION
# ================================================================
echo ""
echo "[3/4] PHASE 3: Quantitative - Computing metrics..."
echo ""

python scripts/evaluation/compute_metrics.py \
    --pred_dir $CKPT_DIR/test \
    --gt_path $OUTPUT_DIR/ghop_gt.pth \
    --output $OUTPUT_DIR/metrics_$TIMESTAMP.json

# ================================================================
# PHASE 4: ANALYSIS & REPORTING
# ================================================================
echo ""
echo "[4/4] PHASE 4: Analysis - Generating report..."
echo ""

python scripts/evaluation/generate_report.py \
    --metrics $OUTPUT_DIR/metrics_$TIMESTAMP.json \
    --visuals $CKPT_DIR/test/visuals \
    --output $OUTPUT_DIR/report_$TIMESTAMP.html

echo ""
echo "================================================================"
echo "EVALUATION COMPLETE!"
echo "================================================================"
echo "Results:"
echo "  Metrics: $OUTPUT_DIR/metrics_$TIMESTAMP.json"
echo "  Report:  $OUTPUT_DIR/report_$TIMESTAMP.html"
echo "  Visuals: $CKPT_DIR/test/visuals"
echo "================================================================"