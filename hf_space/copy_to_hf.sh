#!/bin/bash
# Helper script to copy necessary files from outputs/ to hf_space/model/
# Run this script from the repository root: ./hf_space/copy_to_hf.sh

set -e  # Exit on error

echo "🗡️ Aletheion HuggingFace Space - File Copy Script"
echo "=================================================="
echo ""

# Define paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_SPACE_DIR="${REPO_ROOT}/hf_space"
MODEL_DIR="${HF_SPACE_DIR}/model"
ASSETS_DIR="${HF_SPACE_DIR}/assets"
OUTPUTS_DIR="${REPO_ROOT}/outputs"
PAPER_DIR="${REPO_ROOT}/paper/en"

# Create directories if they don't exist
mkdir -p "${MODEL_DIR}"
mkdir -p "${ASSETS_DIR}"

echo "📁 Repository root: ${REPO_ROOT}"
echo "📂 HF Space directory: ${HF_SPACE_DIR}"
echo ""

# Function to copy file if it exists
copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "✅ Copied: $src -> $dst"
        return 0
    else
        echo "⚠️  Not found: $src"
        return 1
    fi
}

# Copy model configuration from pyramidal output
echo "📋 Copying model configuration..."
if copy_if_exists "${OUTPUTS_DIR}/pyramidal/config.json" "${MODEL_DIR}/config.json"; then
    echo "   Model config copied successfully"
else
    echo "   Creating default config.json..."
    cat > "${MODEL_DIR}/config.json" << 'EOF'
{
  "steps": null,
  "num_epochs": 10,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "gradient_checkpointing": true,
  "fp16": true,
  "lr": 0.0003,
  "lambda_base": 0.005,
  "lambda_height": 0.02,
  "height_method": "error_based",
  "multi_head_height": false,
  "no_temp_modulation": false,
  "eval_interval": 500,
  "save_interval": 500,
  "seed": 42,
  "dry_run": false,
  "resume_from": null,
  "output_dir": "outputs/pyramidal"
}
EOF
    echo "   ✅ Created default config.json"
fi

# Copy model weights (check multiple possible locations)
echo ""
echo "🔍 Searching for model weights..."
FOUND_WEIGHTS=false

# Check for weights in various possible locations
WEIGHT_LOCATIONS=(
    "${OUTPUTS_DIR}/pyramidal/pytorch_model.bin"
    "${OUTPUTS_DIR}/pyramidal/model.bin"
    "${OUTPUTS_DIR}/pyramidal/checkpoint_final.pt"
    "${OUTPUTS_DIR}/pyramidal/final/pytorch_model.bin"
    "${OUTPUTS_DIR}/baseline/final/pytorch_model.bin"
)

for weight_path in "${WEIGHT_LOCATIONS[@]}"; do
    if [ -f "$weight_path" ]; then
        cp "$weight_path" "${MODEL_DIR}/pytorch_model.bin"
        echo "✅ Found and copied weights: $weight_path"
        FOUND_WEIGHTS=true
        break
    fi
done

if [ "$FOUND_WEIGHTS" = false ]; then
    echo "⚠️  No model weights found. The demo will use randomly initialized weights."
    echo "   Train a model first using: python examples/train_aletheion.py"
    echo "   Or copy weights manually to: ${MODEL_DIR}/pytorch_model.bin"
fi

# Copy GPT-2 tokenizer files (will be downloaded at runtime if not present)
echo ""
echo "📝 Tokenizer setup..."
echo "   Tokenizer will be downloaded from HuggingFace at runtime (gpt2)"

# Copy training curves
echo ""
echo "📊 Copying training visualizations..."
copy_if_exists "${OUTPUTS_DIR}/pyramidal/training_curves.png" "${ASSETS_DIR}/training_curves.png"
copy_if_exists "${OUTPUTS_DIR}/comparison/perplexity_ece_scatter.png" "${ASSETS_DIR}/comparison_plot.png"
copy_if_exists "${OUTPUTS_DIR}/comparison/calibration_plots.png" "${ASSETS_DIR}/calibration_plots.png"

# Copy paper
echo ""
echo "📄 Copying paper..."
if copy_if_exists "${PAPER_DIR}/aletheion_paper_v5.pdf" "${ASSETS_DIR}/paper.pdf"; then
    echo "   Paper copied successfully"
else
    echo "   ⚠️  Paper not found at expected location"
fi

# Copy source files needed for the demo
echo ""
echo "💻 Copying source files..."
SRC_DIR="${HF_SPACE_DIR}/src"
mkdir -p "${SRC_DIR}/aletheion"

# Copy essential model source files
cp "${REPO_ROOT}/src/model.py" "${SRC_DIR}/model.py"
cp "${REPO_ROOT}/src/attention.py" "${SRC_DIR}/attention.py"
cp -r "${REPO_ROOT}/src/aletheion/"* "${SRC_DIR}/aletheion/"
echo "✅ Source files copied"

# Create __init__.py files
touch "${SRC_DIR}/__init__.py"
if [ ! -f "${SRC_DIR}/aletheion/__init__.py" ]; then
    cp "${REPO_ROOT}/src/aletheion/__init__.py" "${SRC_DIR}/aletheion/__init__.py"
fi

# Summary
echo ""
echo "=================================================="
echo "✨ File copying complete!"
echo ""
echo "📦 HuggingFace Space structure:"
echo "   hf_space/"
echo "   ├── app.py                  ✅"
echo "   ├── requirements.txt        ✅"
echo "   ├── README.md               ✅"
echo "   ├── .gitattributes          ✅"
echo "   ├── model/"
echo "   │   ├── config.json         $([ -f "${MODEL_DIR}/config.json" ] && echo "✅" || echo "❌")"
echo "   │   └── pytorch_model.bin   $([ -f "${MODEL_DIR}/pytorch_model.bin" ] && echo "✅" || echo "⚠️  (optional)")"
echo "   ├── assets/"
echo "   │   ├── paper.pdf           $([ -f "${ASSETS_DIR}/paper.pdf" ] && echo "✅" || echo "⚠️  (optional)")"
echo "   │   └── training_curves.png $([ -f "${ASSETS_DIR}/training_curves.png" ] && echo "✅" || echo "⚠️  (optional)")"
echo "   └── src/                    ✅"
echo ""
echo "🚀 Next steps:"
echo "   1. cd hf_space"
echo "   2. git init"
echo "   3. git lfs install"
echo "   4. git remote add origin https://huggingface.co/spaces/gnai-creator/aletheion-llm"
echo "   5. git add ."
echo "   6. git commit -m 'Initial Aletheion Space deployment'"
echo "   7. git push -u origin main"
echo ""
echo "📖 Documentation: https://huggingface.co/docs/hub/spaces"
echo ""
