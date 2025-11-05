# HuggingFace Space Setup - Complete Summary

## 🗡️ Aletheion LLM Demo - Ready for Deployment

This directory contains a complete, deployment-ready HuggingFace Space for the Aletheion Pyramidal Model.

---

## 📦 What's Included

### Core Files

| File | Description | Status |
|------|-------------|--------|
| `app.py` | Gradio interface with text generation + uncertainty metrics | ✅ Ready |
| `requirements.txt` | Python dependencies (PyTorch CPU, Gradio 4.x, etc.) | ✅ Ready |
| `README.md` | HuggingFace Space card with YAML frontmatter | ✅ Ready |
| `.gitattributes` | Git LFS configuration for large files | ✅ Ready |

### Model Files

| File | Description | Status |
|------|-------------|--------|
| `model/config.json` | Pyramidal model configuration | ✅ Included |
| `model/pytorch_model.bin` | Trained model weights | ⚠️ Not included (optional) |

> **Note:** Demo will use randomly initialized weights if pytorch_model.bin is not present. Add trained weights for better results.

### Assets

| File | Description | Status |
|------|-------------|--------|
| `assets/paper.pdf` | Research paper (Geometry of Knowing) | ✅ Included |
| `assets/training_curves.png` | Training visualizations | ✅ Included |
| `assets/comparison_plot.png` | Perplexity vs ECE comparison | ✅ Included |
| `assets/calibration_plots.png` | Calibration plots | ✅ Included |

### Source Code

| Directory/File | Description | Status |
|----------------|-------------|--------|
| `src/model.py` | Baseline transformer implementation | ✅ Included |
| `src/attention.py` | Attention mechanisms | ✅ Included |
| `src/aletheion/` | Aletheion epistemic modules | ✅ Included |
| `src/aletheion/pyramidal_model.py` | Pyramidal transformer | ✅ Included |
| `src/aletheion/gates.py` | Epistemic gates (Q₁, Q₂) | ✅ Included |
| `src/aletheion/loss.py` | VARO loss implementation | ✅ Included |
| `src/aletheion/pyramid.py` | Pyramidal geometry logic | ✅ Included |

### Helper Files

| File | Description | Status |
|------|-------------|--------|
| `copy_to_hf.sh` | Script to copy files from outputs/ | ✅ Included |
| `test_local.py` | Local testing script (run before deployment) | ✅ Included |
| `DEPLOYMENT.md` | Comprehensive deployment guide | ✅ Included |

---

## 🎯 Key Features Implemented

### 1. Interactive Gradio Interface

- **Text Generation**: Enter prompt, adjust parameters, generate text
- **Uncertainty Metrics**: Display height, base stability, uncertainty, confidence, ECE
- **Example Prompts**: Pre-configured examples showcasing different scenarios
- **Parameter Controls**:
  - Max length (10-200 tokens)
  - Temperature (0.1-2.0)
  - Top-k sampling (0-100)
  - Top-p/nucleus sampling (0.0-1.0)

### 2. Epistemic Uncertainty Quantification

The demo displays these metrics for each generation:

- **Height**: Proximity to truth apex (0 = uncertain, 1 = certain)
- **Base Stability**: Consistency of epistemic forces (Memory, Pain, Choice, Exploration)
- **Uncertainty**: `1 - height` (explicit uncertainty measure)
- **Confidence**: `height × base_stability` (combined certainty)
- **ECE (estimated)**: Approximate Expected Calibration Error

### 3. Pyramidal Architecture

Implements the 5-vertex geometric structure:

```
        Truth (Apex = 1.0)
           ▲
          /|\
         / | \
        /  h \      h = height (distance from base to apex)
       /   |   \
      /    |    \
     /_____+_____\
    Base Forces (4 vertices):
    - w₁: Memory
    - w₂: Pain (Loss)
    - w₃: Choice (Entropy)
    - w₄: Exploration (KL)
```

### 4. Model Compatibility

- **Base Model**: GPT-2 architecture (6 layers, 512d, 8 heads)
- **Tokenizer**: GPT-2 tokenizer (50,257 vocab)
- **CPU Optimized**: Works on HuggingFace free tier
- **Graceful Degradation**: Works with or without trained weights

---

## 🚀 Quick Start

### Option 1: Local Testing

```bash
# Navigate to hf_space directory
cd hf_space

# Run local tests
python test_local.py

# Test the Gradio interface locally
python app.py
# Open browser to http://localhost:7860
```

### Option 2: Deploy to HuggingFace

```bash
# Follow detailed instructions in DEPLOYMENT.md

# Quick version:
cd hf_space
git init
git lfs install
git remote add origin https://huggingface.co/spaces/USERNAME/aletheion-llm
git add .
git commit -m "Initial Aletheion Space deployment"
git push -u origin main
```

---

## 📊 Expected Results

### Benchmark Performance

Based on training results:

| Metric | Baseline | Aletheion Pyramidal | Improvement |
|--------|----------|---------------------|-------------|
| **ECE** | 0.104 | 0.011 | **89% reduction** |
| Perplexity | ~45 | ~43 | ~4% better |
| Parameters | 42M | 43M | +2% overhead |

### Demo Behavior

**With trained weights:**
- Generates coherent text
- Displays meaningful uncertainty metrics
- Shows calibrated confidence scores

**Without trained weights (current):**
- Generates random/nonsensical text (expected)
- Still demonstrates the interface
- Shows how uncertainty metrics work

---

## 🔧 Customization Guide

### Add Trained Weights

```bash
# Copy your trained checkpoint
cp /path/to/checkpoint.pt hf_space/model/pytorch_model.bin

# Or re-run the copy script
./hf_space/copy_to_hf.sh

# Commit and push
cd hf_space
git add model/pytorch_model.bin
git commit -m "Add trained model weights"
git push
```

### Modify Interface

Edit `app.py`:

```python
# Change default parameters
temperature = gr.Slider(value=0.9)  # Default temp

# Add custom examples
examples = [
    ["Your custom prompt", 50, 1.0, 50, 0.95],
]

# Modify metrics display
def format_metrics(metrics):
    # Custom formatting logic
    pass
```

### Update Space Card

Edit `README.md` frontmatter:

```yaml
---
title: Your Custom Title
emoji: 🗡️  # or any emoji
colorFrom: blue
colorTo: purple
pinned: true  # Pin to your profile
---
```

---

## 📋 Pre-Deployment Checklist

- [x] ✅ `app.py` created with Gradio interface
- [x] ✅ `requirements.txt` with all dependencies
- [x] ✅ `README.md` with HF Space frontmatter
- [x] ✅ `.gitattributes` for Git LFS
- [x] ✅ Model config copied
- [x] ✅ Source code copied (src/)
- [x] ✅ Assets copied (paper, plots)
- [x] ✅ Helper scripts created
- [ ] ⚠️ Model weights (optional - add if available)
- [ ] 🚀 Deploy to HuggingFace (see DEPLOYMENT.md)

---

## 🔗 Important Links

### Documentation
- **Deployment Guide**: `DEPLOYMENT.md`
- **Main README**: `README.md` (Space card)
- **Test Script**: `test_local.py`

### Resources
- **Paper**: `assets/paper.pdf`
- **Training Curves**: `assets/training_curves.png`
- **Comparisons**: `assets/comparison_plot.png`

### External
- **GitHub**: https://github.com/AletheionAGI/aletheion-llm
- **HuggingFace Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://www.gradio.app/docs

---

## 🎓 Technical Details

### Architecture Highlights

1. **Pyramidal Epistemic Gates**: Output-only gating (Level 1)
2. **Height-Based Temperature Modulation**: Adjusts confidence by uncertainty
3. **VARO Loss**: `L = L_CE + λ_base·L_base + λ_height·L_height`
4. **Base Forces**: Memory, Pain (loss), Choice (entropy), Exploration (KL)

### Model Parameters

```json
{
  "vocab_size": 50257,
  "d_model": 512,
  "n_layers": 6,
  "n_heads": 8,
  "d_ff": 2048,
  "max_seq_len": 512,
  "lambda_base": 0.005,
  "lambda_height": 0.02,
  "height_method": "error_based"
}
```

### Generation Pipeline

1. **Tokenize** input prompt
2. **Forward pass** through model → logits + pyramid metrics
3. **Sample** next token (with temp/top-k/top-p)
4. **Collect** height, base_stability per token
5. **Aggregate** metrics over sequence
6. **Display** generated text + uncertainty

---

## 🐛 Known Limitations

1. **No Trained Weights**: Demo uses random weights (add checkpoint to fix)
2. **CPU Only**: Slower inference on free tier (upgrade to GPU tier)
3. **Simplified ECE**: Rough approximation (proper ECE needs test set)
4. **Short Context**: Max 512 tokens (model limitation)
5. **Q1Q2 Not Included**: Level 2 architecture coming soon

---

## 🎉 Success Criteria

Your Space is ready for deployment when:

- [x] All core files present (app.py, requirements.txt, README.md)
- [x] Source code copied to src/
- [x] Assets available (paper, plots)
- [x] Config JSON valid
- [x] Local testing passes (`python test_local.py`)
- [ ] (Optional) Trained weights added
- [ ] Git repository initialized
- [ ] Pushed to HuggingFace

---

## 📧 Support

- **Issues**: https://github.com/AletheionAGI/aletheion-llm/issues
- **Email**: contact@alethea.tech
- **Discord**: .lacivo

---

**Made with ❤️ by the Aletheion team**

*Climbing toward truth, one token at a time.* 🗡️
