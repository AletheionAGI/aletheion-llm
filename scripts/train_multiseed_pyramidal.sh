#!/bin/bash
# Multi-Seed Training Script for Pyramidal Q1/Q2 Model
#
# This script trains the Pyramidal Q1/Q2 model with multiple random seeds
# for statistical validation and reproducibility testing.
#
# Usage:
#   ./scripts/train_multiseed_pyramidal.sh [config_file] [output_base_dir]
#
# Example:
#   ./scripts/train_multiseed_pyramidal.sh config/aletheion_level1.yaml outputs/validation
#
# The script will:
# - Train with 5 different seeds (42, 123, 456, 789, 999)
# - Generate comprehensive Markdown reports for each run
# - Save all models, histories, and training curves
# - Create individual reports: TRAINING_REPORT_seed_XX.md
#
# Expected output structure:
#   outputs/validation/
#     ├── seed_42/
#     │   ├── TRAINING_REPORT_seed_42.md
#     │   ├── training_curves.png
#     │   ├── history.json
#     │   └── final_model/
#     ├── seed_123/
#     │   ├── TRAINING_REPORT_seed_123.md
#     │   └── ...
#     └── ...

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${1:-config/aletheion_level1.yaml}"
OUTPUT_BASE="${2:-outputs/validation}"
SEEDS=(42 123 456 789 999)

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Multi-Seed Pyramidal Q1/Q2 Training Script             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Config file:    ${CONFIG_FILE}"
echo -e "  Output base:    ${OUTPUT_BASE}"
echo -e "  Seeds:          ${SEEDS[*]}"
echo -e "  Total runs:     ${#SEEDS[@]}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Error: Config file not found: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}Creating default config...${NC}"
    mkdir -p "$(dirname "$CONFIG_FILE")"
    # Note: You would need to create a default config here or provide one
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Log file for all runs
MASTER_LOG="$OUTPUT_BASE/multiseed_training.log"
echo "Multi-seed training started at $(date)" > "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Track successful and failed runs
SUCCESSFUL_RUNS=()
FAILED_RUNS=()

# Main training loop
for seed in "${SEEDS[@]}"; do
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Training with seed: ${seed}                                 ${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    OUTPUT_DIR="$OUTPUT_BASE/seed_$seed"
    LOG_FILE="$OUTPUT_DIR/training.log"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Log start time
    START_TIME=$(date +%s)
    echo -e "${YELLOW}▶ Started at: $(date)${NC}"

    # Run training
    echo "Training seed $seed..." >> "$MASTER_LOG"

    if python experiments/level1/train_pyramidal_q1q2.py \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        --seed "$seed" \
        --wandb_run_name "aletheion_level1_seed${seed}" \
        > "$LOG_FILE" 2>&1; then

        # Training succeeded
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        HOURS=$((ELAPSED / 3600))
        MINUTES=$(((ELAPSED % 3600) / 60))
        SECONDS=$((ELAPSED % 60))

        echo -e "${GREEN}✅ Seed $seed completed successfully!${NC}"
        echo -e "${GREEN}   Time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
        echo "  ✓ Completed in ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MASTER_LOG"

        SUCCESSFUL_RUNS+=("$seed")

        # Check if report was generated
        REPORT_FILE="$OUTPUT_DIR/TRAINING_REPORT_seed_${seed}.md"
        if [ -f "$REPORT_FILE" ]; then
            echo -e "${GREEN}   Report: ${REPORT_FILE}${NC}"
        else
            echo -e "${YELLOW}   ⚠️  Warning: Report not found${NC}"
        fi

    else
        # Training failed
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        echo -e "${RED}❌ Seed $seed failed!${NC}"
        echo -e "${RED}   Check log: ${LOG_FILE}${NC}"
        echo "  ✗ Failed after ${ELAPSED}s" >> "$MASTER_LOG"
        echo "    Error: Check $LOG_FILE for details" >> "$MASTER_LOG"

        FAILED_RUNS+=("$seed")
    fi

    echo "" >> "$MASTER_LOG"
done

# Print summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    SUMMARY                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Successful runs: ${#SUCCESSFUL_RUNS[@]}/${#SEEDS[@]}${NC}"
if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo -e "  Seeds: ${SUCCESSFUL_RUNS[*]}"
fi

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo -e "${RED}Failed runs: ${#FAILED_RUNS[@]}/${#SEEDS[@]}${NC}"
    echo -e "  Seeds: ${FAILED_RUNS[*]}"
fi

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review individual reports: ${OUTPUT_BASE}/seed_*/TRAINING_REPORT_seed_*.md"
echo -e "  2. Analyze multi-seed statistics:"
echo -e "     ${BLUE}python scripts/analyze_multiseed.py${NC}"
echo -e "  3. Generate reliability diagrams"
echo -e "  4. Update documentation with results"
echo ""

# Save summary
SUMMARY_FILE="$OUTPUT_BASE/MULTISEED_SUMMARY.md"
cat > "$SUMMARY_FILE" << EOF
# Multi-Seed Training Summary

**Date:** $(date)
**Config:** $CONFIG_FILE
**Output:** $OUTPUT_BASE

## Results

| Seed | Status | Report | Training Curves |
|------|--------|--------|------------------|
EOF

for seed in "${SEEDS[@]}"; do
    if [[ " ${SUCCESSFUL_RUNS[*]} " =~ " ${seed} " ]]; then
        STATUS="✅ Success"
        REPORT="[View](seed_${seed}/TRAINING_REPORT_seed_${seed}.md)"
        CURVES="[View](seed_${seed}/training_curves.png)"
    else
        STATUS="❌ Failed"
        REPORT="N/A"
        CURVES="N/A"
    fi

    echo "| $seed | $STATUS | $REPORT | $CURVES |" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

## Statistics

- **Total runs:** ${#SEEDS[@]}
- **Successful:** ${#SUCCESSFUL_RUNS[@]}
- **Failed:** ${#FAILED_RUNS[@]}
- **Success rate:** $(( ${#SUCCESSFUL_RUNS[@]} * 100 / ${#SEEDS[@]} ))%

## Next Steps

1. **Statistical Analysis:**
   \`\`\`bash
   python scripts/analyze_multiseed.py
   \`\`\`

2. **Review Individual Reports:**
   - Each seed has a detailed report in \`seed_XX/TRAINING_REPORT_seed_XX.md\`
   - Check metrics: ECE, Perplexity, Brier Score, Q1/Q2, etc.

3. **Generate Comparative Visualizations:**
   - Compare Q1/Q2 convergence across seeds
   - Analyze calibration metric distributions
   - Verify reproducibility (coefficient of variation)

4. **Documentation:**
   - Update README.md with multi-seed validation results
   - Add results to QUANTITATIVE_METRICS_ANALYSIS.md
   - Create LEVEL1_VALIDATION_COMPLETE.md if all metrics pass

## Files Generated

\`\`\`
$OUTPUT_BASE/
├── MULTISEED_SUMMARY.md (this file)
├── multiseed_training.log
$(for seed in "${SUCCESSFUL_RUNS[@]}"; do
    echo "├── seed_$seed/"
    echo "│   ├── TRAINING_REPORT_seed_${seed}.md"
    echo "│   ├── training_curves.png"
    echo "│   ├── history.json"
    echo "│   ├── config.json"
    echo "│   └── final_model/"
done)
\`\`\`

---

**Generated:** $(date)
EOF

echo -e "${GREEN}✅ Summary saved to: ${SUMMARY_FILE}${NC}"
echo ""

# Exit with appropriate code
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
