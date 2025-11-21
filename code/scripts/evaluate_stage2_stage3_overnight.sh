#!/bin/bash
# ================================================================
# HOLDSE Stage 2 & 3 Complete Evaluation Pipeline
# Automated overnight run - no manual intervention needed
# Total duration: ~7 hours
# ================================================================

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# ================================================================
# Configuration
# ================================================================
CODE_DIR="$HOME/Projects/holdse/code"
EVAL_DIR="${CODE_DIR}/logs/evaluation_results"
STAGE2_CKPT="${CODE_DIR}/logs/stage2_final.ckpt"
STAGE3_CKPT="${CODE_DIR}/logs/stage3_final.ckpt"
HOLD_BASELINE="$HOME/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json"

# Ensure eval directory exists
mkdir -p "${EVAL_DIR}"

# Log file for entire run
MASTER_LOG="${EVAL_DIR}/overnight_evaluation_$(date +%Y%m%d_%H%M%S).log"

# ================================================================
# Utility Functions
# ================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MASTER_LOG}"
}

log_section() {
    echo "" | tee -a "${MASTER_LOG}"
    echo "================================================================" | tee -a "${MASTER_LOG}"
    echo "$1" | tee -a "${MASTER_LOG}"
    echo "================================================================" | tee -a "${MASTER_LOG}"
    echo "" | tee -a "${MASTER_LOG}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log "❌ ERROR: Required file not found: $1"
        exit 1
    fi
    log "✓ Found: $1"
}

estimate_time() {
    local start_time=$1
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    log "⏱️  Time elapsed: ${hours}h ${minutes}m"
}

# ================================================================
# Pre-flight Checks
# ================================================================

log_section "STARTING OVERNIGHT EVALUATION PIPELINE"

log "🔍 Pre-flight checks..."

# Check checkpoints exist
check_file_exists "${STAGE2_CKPT}"
check_file_exists "${STAGE3_CKPT}"

# Check scripts exist
check_file_exists "${CODE_DIR}/scripts/extract_predictions.py"
check_file_exists "${CODE_DIR}/scripts/evaluate_predictions.py"

# Check baseline exists (optional, warn if missing)
if [ ! -f "${HOLD_BASELINE}" ]; then
    log "⚠️  WARNING: HOLD baseline not found, comparisons will be skipped"
    HOLD_BASELINE=""
fi

log "✅ All pre-flight checks passed"
log ""
log "📊 Pipeline schedule:"
log "  1. Stage 2 extraction:  ~3.5 hours"
log "  2. Stage 2 evaluation:  ~30 seconds"
log "  3. Stage 3 extraction:  ~3.5 hours"
log "  4. Stage 3 evaluation:  ~30 seconds"
log "  5. Comparison report:   ~10 seconds"
log "  Total duration:         ~7 hours"
log ""
log "💤 You can sleep now. Check results in the morning!"

PIPELINE_START=$(date +%s)

# ================================================================
# STAGE 2: EXTRACTION
# ================================================================

log_section "STAGE 2: EXTRACTION"

STAGE2_START=$(date +%s)

log "🚀 Starting Stage 2 extraction..."
log "   Checkpoint: ${STAGE2_CKPT}"
log "   Output: ${EVAL_DIR}/MC1_stage2_predictions.pkl"
log "   Expected duration: 3.5 hours (88 seconds × 144 frames)"

python "${CODE_DIR}/scripts/extract_predictions.py" \
    --checkpoint "${STAGE2_CKPT}" \
    --seq_name hold_MC1_ho3d \
    --config "${CODE_DIR}/confs/ghop_stage2_temporal_only.yaml" \
    --output "${EVAL_DIR}/MC1_stage2_predictions.pkl" \
    2>&1 | tee -a "${EVAL_DIR}/MC1_stage2_extraction.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ ERROR: Stage 2 extraction failed!"
    exit 1
fi

STAGE2_EXTRACT_END=$(date +%s)
estimate_time ${STAGE2_START}

# Verify output exists
if [ ! -f "${EVAL_DIR}/MC1_stage2_predictions.pkl" ]; then
    log "❌ ERROR: Stage 2 predictions file not created!"
    exit 1
fi

FILESIZE=$(du -h "${EVAL_DIR}/MC1_stage2_predictions.pkl" | cut -f1)
log "✅ Stage 2 extraction complete! File size: ${FILESIZE}"

# ================================================================
# STAGE 2: EVALUATION
# ================================================================

log_section "STAGE 2: EVALUATION"

log "📊 Evaluating Stage 2 predictions..."

if [ -n "${HOLD_BASELINE}" ]; then
    python "${CODE_DIR}/scripts/evaluate_predictions.py" \
        --predictions "${EVAL_DIR}/MC1_stage2_predictions.pkl" \
        --compare "${HOLD_BASELINE}" \
        2>&1 | tee "${EVAL_DIR}/MC1_stage2_evaluation.txt"
else
    python "${CODE_DIR}/scripts/evaluate_predictions.py" \
        --predictions "${EVAL_DIR}/MC1_stage2_predictions.pkl" \
        2>&1 | tee "${EVAL_DIR}/MC1_stage2_evaluation.txt"
fi

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ ERROR: Stage 2 evaluation failed!"
    exit 1
fi

log "✅ Stage 2 evaluation complete!"

# Show quick summary
if [ -f "${EVAL_DIR}/MC1_stage2_predictions.pkl.metric.json" ]; then
    log ""
    log "📈 Stage 2 Quick Results:"
    python << 'EOF'
import json
import sys
try:
    with open('logs/evaluation_results/MC1_stage2_predictions.pkl.metric.json') as f:
        m = json.load(f)
    print(f"  MPJPE: {m.get('mpjpe_ra_r', 'N/A'):.2f} mm")
    print(f"  CD_RA: {m.get('cd_ra', 'N/A'):.2f} cm²")
    print(f"  F10_RA: {m.get('f10_ra', 'N/A'):.2f}%")
except Exception as e:
    print(f"  Could not parse metrics: {e}")
EOF
fi

estimate_time ${PIPELINE_START}

# ================================================================
# STAGE 3: EXTRACTION
# ================================================================

log_section "STAGE 3: EXTRACTION"

STAGE3_START=$(date +%s)

log "🚀 Starting Stage 3 extraction..."
log "   Checkpoint: ${STAGE3_CKPT}"
log "   Output: ${EVAL_DIR}/MC1_stage3_predictions.pkl"
log "   Expected duration: 3.5 hours (88 seconds × 144 frames)"

python "${CODE_DIR}/scripts/extract_predictions.py" \
    --checkpoint "${STAGE3_CKPT}" \
    --seq_name hold_MC1_ho3d \
    --config "${CODE_DIR}/confs/ghop_stage3_full_pipeline.yaml" \
    --output "${EVAL_DIR}/MC1_stage3_predictions.pkl" \
    2>&1 | tee -a "${EVAL_DIR}/MC1_stage3_extraction.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ ERROR: Stage 3 extraction failed!"
    exit 1
fi

STAGE3_EXTRACT_END=$(date +%s)
estimate_time ${STAGE3_START}

# Verify output exists
if [ ! -f "${EVAL_DIR}/MC1_stage3_predictions.pkl" ]; then
    log "❌ ERROR: Stage 3 predictions file not created!"
    exit 1
fi

FILESIZE=$(du -h "${EVAL_DIR}/MC1_stage3_predictions.pkl" | cut -f1)
log "✅ Stage 3 extraction complete! File size: ${FILESIZE}"

# ================================================================
# STAGE 3: EVALUATION
# ================================================================

log_section "STAGE 3: EVALUATION"

log "📊 Evaluating Stage 3 predictions..."

if [ -n "${HOLD_BASELINE}" ]; then
    python "${CODE_DIR}/scripts/evaluate_predictions.py" \
        --predictions "${EVAL_DIR}/MC1_stage3_predictions.pkl" \
        --compare "${HOLD_BASELINE}" \
        2>&1 | tee "${EVAL_DIR}/MC1_stage3_evaluation.txt"
else
    python "${CODE_DIR}/scripts/evaluate_predictions.py" \
        --predictions "${EVAL_DIR}/MC1_stage3_predictions.pkl" \
        2>&1 | tee "${EVAL_DIR}/MC1_stage3_evaluation.txt"
fi

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ ERROR: Stage 3 evaluation failed!"
    exit 1
fi

log "✅ Stage 3 evaluation complete!"

# Show quick summary
if [ -f "${EVAL_DIR}/MC1_stage3_predictions.pkl.metric.json" ]; then
    log ""
    log "📈 Stage 3 Quick Results:"
    python << 'EOF'
import json
import sys
try:
    with open('logs/evaluation_results/MC1_stage3_predictions.pkl.metric.json') as f:
        m = json.load(f)
    print(f"  MPJPE: {m.get('mpjpe_ra_r', 'N/A'):.2f} mm")
    print(f"  CD_RA: {m.get('cd_ra', 'N/A'):.2f} cm²")
    print(f"  F10_RA: {m.get('f10_ra', 'N/A'):.2f}%")
except Exception as e:
    print(f"  Could not parse metrics: {e}")
EOF
fi

# ================================================================
# COMPARISON REPORT
# ================================================================

log_section "GENERATING COMPARISON REPORT"

log "📊 Creating Stage 1 → 2 → 3 comparison..."

python << 'EOF' | tee "${EVAL_DIR}/MC1_complete_comparison.txt"
import json
import os

print("\n" + "="*85)
print("HOLDSE COMPLETE PIPELINE EVALUATION: Stage 1 → 2 → 3")
print("="*85)

# Load all three stages
stages = {}
for stage_name in ['MC1_stage1_e200', 'MC1_stage2', 'MC1_stage3']:
    metric_file = f'logs/evaluation_results/{stage_name}_predictions.pkl.metric.json'
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            stages[stage_name] = json.load(f)
    else:
        print(f"⚠️  WARNING: {stage_name} metrics not found")
        stages[stage_name] = None

# Check if we have all stages
if not all(stages.values()):
    print("\n❌ Cannot generate comparison - missing stage results")
    exit(1)

# Comparison table
print(f"\n{'Metric':<15} {'Stage 1':>12} {'Stage 2':>12} {'Stage 3':>12} {'S1→S2':>12} {'S2→S3':>12} {'Total':>12}")
print("-"*85)

metrics_to_compare = [
    ('mpjpe_ra_r', 'mm', False),
    ('mrrpe_ho', 'mm', False),
    ('cd_ra', 'cm²', False),
    ('f10_ra', '%', True),
    ('f5_ra', '%', True)
]

for metric, unit, higher_better in metrics_to_compare:
    s1 = stages['MC1_stage1_e200'].get(metric)
    s2 = stages['MC1_stage2'].get(metric)
    s3 = stages['MC1_stage3'].get(metric)
    
    if s1 is not None and s2 is not None and s3 is not None:
        s1_s2 = s2 - s1
        s2_s3 = s3 - s2
        total = s3 - s1
        
        # Determine if improvements are good
        s1_s2_good = (s1_s2 < 0 and not higher_better) or (s1_s2 > 0 and higher_better)
        s2_s3_good = (s2_s3 < 0 and not higher_better) or (s2_s3 > 0 and higher_better)
        total_good = (total < 0 and not higher_better) or (total > 0 and higher_better)
        
        s1_s2_ind = "✅" if s1_s2_good else "❌"
        s2_s3_ind = "✅" if s2_s3_good else "❌"
        total_ind = "✅" if total_good else "❌"
        
        print(f"{metric:<15} {s1:>12.2f} {s2:>12.2f} {s3:>12.2f} "
              f"{s1_s2:>+11.2f}{s1_s2_ind} {s2_s3:>+11.2f}{s2_s3_ind} {total:>+11.2f}{total_ind}")

print("="*85)

# Analysis
print("\n📊 REFINEMENT EFFECTIVENESS ANALYSIS\n")

s1_mpjpe = stages['MC1_stage1_e200']['mpjpe_ra_r']
s2_mpjpe = stages['MC1_stage2']['mpjpe_ra_r']
s3_mpjpe = stages['MC1_stage3']['mpjpe_ra_r']

stage2_improvement = s1_mpjpe - s2_mpjpe
stage3_additional = s2_mpjpe - s3_mpjpe
total_improvement = s1_mpjpe - s3_mpjpe

print(f"Stage 1 → 2 (GHOP SDS):              {stage2_improvement:+.2f} mm ({stage2_improvement/s1_mpjpe*100:+.1f}%)")
print(f"Stage 2 → 3 (Contact + Temporal):   {stage3_additional:+.2f} mm ({stage3_additional/s2_mpjpe*100:+.1f}%)")
print(f"Total pipeline improvement:          {total_improvement:+.2f} mm ({total_improvement/s1_mpjpe*100:+.1f}%)")

print(f"\nFinal Stage 3 MPJPE: {s3_mpjpe:.2f} mm")

# Decision recommendation
print("\n" + "="*85)
print("RECOMMENDATION")
print("="*85)

if total_improvement > 15:
    print("\n✅ STRONG IMPROVEMENT (>15mm total)")
    print("   → Refinement pipeline is EFFECTIVE")
    print("   → Recommended next step: Train Stage 2/3 on HOLD checkpoint (cb20a1702)")
    print("   → Expected final result: 18-22mm MPJPE (state-of-the-art)")
elif total_improvement > 8:
    print("\n⚠️  MODERATE IMPROVEMENT (8-15mm total)")
    print("   → Refinements help but are limited by Stage 1 quality")
    print("   → Recommended next step: Fix Stage 1 architecture, then retrain pipeline")
    print("   → Expected final result: 28-32mm MPJPE (good quality)")
else:
    print("\n❌ MINIMAL IMPROVEMENT (<8mm total)")
    print("   → Refinements not effective on poor baseline")
    print("   → Recommended next step: Investigate refinement pipeline issues")
    print("   → Consider: Train Stage 2/3 on HOLD checkpoint to validate refinements")

print("\n" + "="*85)
EOF

if [ $? -ne 0 ]; then
    log "⚠️  Warning: Comparison report generation failed, but evaluations completed successfully"
fi

# ================================================================
# PIPELINE COMPLETE
# ================================================================

PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

log_section "OVERNIGHT EVALUATION COMPLETE! ✅"

log "⏱️  Total pipeline duration: ${HOURS}h ${MINUTES}m"
log ""
log "📁 Generated files:"
log "   ${EVAL_DIR}/MC1_stage2_predictions.pkl"
log "   ${EVAL_DIR}/MC1_stage2_predictions.pkl.metric.json"
log "   ${EVAL_DIR}/MC1_stage2_evaluation.txt"
log "   ${EVAL_DIR}/MC1_stage3_predictions.pkl"
log "   ${EVAL_DIR}/MC1_stage3_predictions.pkl.metric.json"
log "   ${EVAL_DIR}/MC1_stage3_evaluation.txt"
log "   ${EVAL_DIR}/MC1_complete_comparison.txt"
log ""
log "📊 View results:"
log "   cat ${EVAL_DIR}/MC1_complete_comparison.txt"
log "   cat ${EVAL_DIR}/MC1_stage2_evaluation.txt"
log "   cat ${EVAL_DIR}/MC1_stage3_evaluation.txt"
log ""
log "🎉 All done! Check the comparison report for next steps."

## Launch in background with nohup
#nohup bash scripts/evaluate_stage2_stage3_overnight.sh > logs/overnight_run.log 2>&1 &
#
## Get process ID
#echo $! > logs/overnight_run.pid
## Watch live progress
#tail -f logs/overnight_run.log
#
## Or watch just the master log
#tail -f logs/evaluation_results/overnight_evaluation_*.log
#
## Check current stage
#grep "STAGE" logs/overnight_run.log | tail -3
#
## Estimate completion time
#grep "Time elapsed" logs/overnight_run.log | tail -1