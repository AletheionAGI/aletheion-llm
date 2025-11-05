#!/bin/bash
# Test models on TruthfulQA dataset

set -e

# Configuration
BASELINE_CHECKPOINT="${1:-outputs/baseline/final}"
PYRAMIDAL_CHECKPOINT="${2:-outputs/pyramidal_q1q2/final}"
OUTPUT_DIR="${3:-outputs/truthfulqa_q1q2}"
MAX_SAMPLES="${4:-200}"

echo "=============================================="
echo "TruthfulQA Evaluation Script (Pyramidal Q1Q2)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Baseline:    $BASELINE_CHECKPOINT"
echo "  Pyramidal:   $PYRAMIDAL_CHECKPOINT (Q1Q2 variant)"
echo "  Output:      $OUTPUT_DIR"
echo "  Max Samples: $MAX_SAMPLES"
echo ""

# Check if checkpoints exist
if [ ! -d "$BASELINE_CHECKPOINT" ]; then
    echo "ERROR: Baseline checkpoint not found at $BASELINE_CHECKPOINT"
    exit 1
fi

if [ ! -d "$PYRAMIDAL_CHECKPOINT" ]; then
    echo "ERROR: Pyramidal Q1Q2 checkpoint not found at $PYRAMIDAL_CHECKPOINT"
    exit 1
fi

# Run evaluation
python experiments/level1/test_truthfulqa.py \
    --baseline "$BASELINE_CHECKPOINT" \
    --pyramidal "$PYRAMIDAL_CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seed 42

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View the report:"
echo "  cat $OUTPUT_DIR/truthfulqa_report.md"
echo ""
echo "View visualizations:"
echo "  ls -lh $OUTPUT_DIR/*.png"
