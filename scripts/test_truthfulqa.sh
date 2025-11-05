#!/bin/bash
# Test models on TruthfulQA dataset
# Supports testing all three model variants: baseline, pyramidal, and pyramidal_q1q2

set -e

# Default configuration
BASELINE_CHECKPOINT="outputs/baseline/final"
MAX_SAMPLES=200

# Parse arguments
MODE="${1:-all}"
SAMPLES_ARG="${2:-}"

# Determine mode and samples
if [[ "$MODE" == "all" ]]; then
    # Test all models
    TEST_PYRAMIDAL=true
    TEST_Q1Q2=true
    if [[ -n "$SAMPLES_ARG" ]]; then
        MAX_SAMPLES="$SAMPLES_ARG"
    fi
elif [[ "$MODE" == "baseline" && "$2" == "pyramidal" ]]; then
    # Test baseline vs pyramidal only
    TEST_PYRAMIDAL=true
    TEST_Q1Q2=false
    if [[ -n "$3" ]]; then
        MAX_SAMPLES="$3"
    fi
elif [[ "$MODE" == "baseline" && "$2" == "pyramidal_q1q2" ]]; then
    # Test baseline vs pyramidal_q1q2 only
    TEST_PYRAMIDAL=false
    TEST_Q1Q2=true
    if [[ -n "$3" ]]; then
        MAX_SAMPLES="$3"
    fi
else
    # Default: test all models
    TEST_PYRAMIDAL=true
    TEST_Q1Q2=true
    # If first argument is a number, treat it as max samples
    if [[ "$MODE" =~ ^[0-9]+$ ]]; then
        MAX_SAMPLES="$MODE"
    fi
fi

echo "============================================================"
echo "TruthfulQA Evaluation Script - Testing All Models"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Baseline:     $BASELINE_CHECKPOINT"
echo "  Max Samples:  $MAX_SAMPLES"
echo ""

# Check if baseline checkpoint exists
if [ ! -d "$BASELINE_CHECKPOINT" ]; then
    echo "ERROR: Baseline checkpoint not found at $BASELINE_CHECKPOINT"
    echo "Please train the baseline model first:"
    echo "  python experiments/level1/train_baseline.py --output outputs/baseline"
    exit 1
fi

# Counter for evaluations
EVAL_COUNT=0
TOTAL_EVALS=0
[[ "$TEST_PYRAMIDAL" == true ]] && ((TOTAL_EVALS++))
[[ "$TEST_Q1Q2" == true ]] && ((TOTAL_EVALS++))

echo "Running $TOTAL_EVALS evaluation(s)..."
echo ""

# Test 1: Baseline vs Pyramidal
if [[ "$TEST_PYRAMIDAL" == true ]]; then
    ((EVAL_COUNT++))
    PYRAMIDAL_CHECKPOINT="outputs/pyramidal/final"
    OUTPUT_DIR="outputs/truthfulqa_pyramidal"

    echo "============================================================"
    echo "Running evaluation $EVAL_COUNT/$TOTAL_EVALS: Baseline vs Pyramidal"
    echo "============================================================"
    echo ""
    echo "Configuration:"
    echo "  Baseline:   $BASELINE_CHECKPOINT"
    echo "  Pyramidal:  $PYRAMIDAL_CHECKPOINT"
    echo "  Output:     $OUTPUT_DIR"
    echo "  Samples:    $MAX_SAMPLES"
    echo ""

    # Check if pyramidal checkpoint exists
    if [ ! -d "$PYRAMIDAL_CHECKPOINT" ]; then
        echo "WARNING: Pyramidal checkpoint not found at $PYRAMIDAL_CHECKPOINT"
        echo "Skipping this evaluation. To train the model:"
        echo "  python experiments/level1/train_pyramidal.py --output outputs/pyramidal"
        echo ""
    else
        # Run evaluation
        python experiments/level1/test_truthfulqa.py \
            --baseline "$BASELINE_CHECKPOINT" \
            --pyramidal "$PYRAMIDAL_CHECKPOINT" \
            --output "$OUTPUT_DIR" \
            --max-samples "$MAX_SAMPLES" \
            --seed 42

        echo ""
        echo "✓ Evaluation $EVAL_COUNT/$TOTAL_EVALS complete!"
        echo "  Results: $OUTPUT_DIR/truthfulqa_report.md"
        echo ""
    fi
fi

# Test 2: Baseline vs Pyramidal Q1Q2
if [[ "$TEST_Q1Q2" == true ]]; then
    ((EVAL_COUNT++))
    PYRAMIDAL_Q1Q2_CHECKPOINT="outputs/pyramidal_q1q2/final"
    OUTPUT_DIR_Q1Q2="outputs/truthfulqa_q1q2"

    echo "============================================================"
    echo "Running evaluation $EVAL_COUNT/$TOTAL_EVALS: Baseline vs Pyramidal Q1Q2"
    echo "============================================================"
    echo ""
    echo "Configuration:"
    echo "  Baseline:      $BASELINE_CHECKPOINT"
    echo "  Pyramidal Q1Q2: $PYRAMIDAL_Q1Q2_CHECKPOINT"
    echo "  Output:        $OUTPUT_DIR_Q1Q2"
    echo "  Samples:       $MAX_SAMPLES"
    echo ""

    # Check if pyramidal_q1q2 checkpoint exists
    if [ ! -d "$PYRAMIDAL_Q1Q2_CHECKPOINT" ]; then
        echo "WARNING: Pyramidal Q1Q2 checkpoint not found at $PYRAMIDAL_Q1Q2_CHECKPOINT"
        echo "Skipping this evaluation. To train the model:"
        echo "  python experiments/level1/train_pyramidal_q1q2.py --output outputs/pyramidal_q1q2"
        echo ""
    else
        # Run evaluation
        python experiments/level1/test_truthfulqa.py \
            --baseline "$BASELINE_CHECKPOINT" \
            --pyramidal "$PYRAMIDAL_Q1Q2_CHECKPOINT" \
            --output "$OUTPUT_DIR_Q1Q2" \
            --max-samples "$MAX_SAMPLES" \
            --seed 42

        echo ""
        echo "✓ Evaluation $EVAL_COUNT/$TOTAL_EVALS complete!"
        echo "  Results: $OUTPUT_DIR_Q1Q2/truthfulqa_report.md"
        echo ""
    fi
fi

echo "============================================================"
echo "All Evaluations Complete!"
echo "============================================================"
echo ""
echo "View results:"
if [[ "$TEST_PYRAMIDAL" == true ]] && [ -d "outputs/truthfulqa_pyramidal" ]; then
    echo "  Baseline vs Pyramidal:"
    echo "    cat outputs/truthfulqa_pyramidal/truthfulqa_report.md"
fi
if [[ "$TEST_Q1Q2" == true ]] && [ -d "outputs/truthfulqa_q1q2" ]; then
    echo "  Baseline vs Pyramidal Q1Q2:"
    echo "    cat outputs/truthfulqa_q1q2/truthfulqa_report.md"
fi
echo ""
