# HuggingFace Space Deployment Instructions

This guide explains how to deploy the Aletheion LLM demo to HuggingFace Spaces.

## 📋 Prerequisites

1. **HuggingFace Account**: Create one at https://huggingface.co/join
2. **Git LFS**: Install Git Large File Storage
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs

   # macOS
   brew install git-lfs

   # Windows
   # Download from https://git-lfs.github.com/
   ```
3. **HuggingFace CLI** (optional but recommended):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

## 🚀 Deployment Steps

### Step 1: Create HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in details:
   - **Name**: `aletheion-llm` (or your preferred name)
   - **License**: `agpl-3.0`
   - **SDK**: `Gradio`
   - **Visibility**: Public or Private
4. Click "Create Space"

### Step 2: Initialize Git Repository

```bash
cd hf_space

# Initialize git repository
git init

# Initialize Git LFS
git lfs install

# Add HuggingFace Space as remote
# Replace USERNAME with your HuggingFace username
git remote add origin https://huggingface.co/spaces/USERNAME/aletheion-llm

# Or use the Space you just created
# git remote add origin https://huggingface.co/spaces/gnai-creator/aletheion-llm
```

### Step 3: Stage Files

```bash
# Add all files
git add .

# Verify LFS is tracking large files
git lfs ls-files
# Should show: assets/paper.pdf, assets/*.png

# Check status
git status
```

### Step 4: Commit and Push

```bash
# Commit all files
git commit -m "Initial Aletheion Space deployment

- Pyramidal epistemic architecture demo
- Gradio interface with uncertainty metrics
- ECE 0.011 vs 0.104 baseline (89% improvement)
- Includes paper, training curves, and visualizations"

# Push to HuggingFace
git push -u origin main

# If prompted, enter your HuggingFace credentials:
# Username: <your-username>
# Password: <your-access-token>  # Get from https://huggingface.co/settings/tokens
```

### Step 5: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/USERNAME/aletheion-llm`
2. You'll see the build logs in the "Logs" tab
3. The Space will automatically build and deploy
4. Once ready, you'll see "Running" status and can interact with the demo

## 📦 What Gets Deployed

```
hf_space/
├── app.py                      # Gradio interface
├── requirements.txt            # Python dependencies
├── README.md                   # Space card (visible on HF)
├── .gitattributes             # LFS configuration
├── model/
│   └── config.json            # Model configuration
├── assets/
│   ├── paper.pdf              # Research paper (LFS)
│   ├── training_curves.png    # Training visualizations (LFS)
│   ├── comparison_plot.png    # Comparison plots (LFS)
│   └── calibration_plots.png  # Calibration plots (LFS)
└── src/                       # Aletheion source code
    ├── model.py
    ├── attention.py
    └── aletheion/
        ├── pyramidal_model.py
        ├── gates.py
        ├── loss.py
        └── ...
```

## 🔧 Adding Model Weights

The current deployment uses randomly initialized weights for demonstration. To add trained model weights:

### Option 1: Copy Checkpoint Manually

```bash
# Copy trained weights to model directory
cp /path/to/your/checkpoint.pt hf_space/model/pytorch_model.bin

# Re-run the copy script to ensure everything is synced
./hf_space/copy_to_hf.sh

# Commit and push
cd hf_space
git add model/pytorch_model.bin
git commit -m "Add trained model weights"
git push
```

### Option 2: Train and Deploy

```bash
# Train the model (from repository root)
python examples/train_aletheion.py --config config/aletheion_level1.yaml

# Copy weights
cp outputs/pyramidal/checkpoint_final.pt hf_space/model/pytorch_model.bin

# Deploy
cd hf_space
git add model/pytorch_model.bin
git commit -m "Add trained Pyramidal model weights"
git push
```

## 🎨 Customization

### Update Space Title/Emoji

Edit `README.md` frontmatter:

```yaml
---
title: Your Custom Title
emoji: 🗡️  # Choose your emoji
colorFrom: blue
colorTo: purple
---
```

### Modify Generation Parameters

Edit `app.py` to change default values:

```python
max_length = gr.Slider(
    minimum=10,
    maximum=200,
    value=100,  # Change default
    # ...
)
```

### Add Custom Examples

Edit the `examples` list in `app.py`:

```python
examples = [
    ["Your custom prompt", 50, 1.0, 50, 0.95],
    # Add more examples
]
```

## 🐛 Troubleshooting

### Build Fails

**Check logs**: Look at the "Logs" tab in your Space for error messages.

Common issues:
- **Missing dependencies**: Add to `requirements.txt`
- **Import errors**: Ensure all source files are in `src/`
- **LFS issues**: Make sure `.gitattributes` is committed

### Out of Memory

HuggingFace free tier has limited resources. To optimize:

1. Use CPU-only PyTorch (already configured in `requirements.txt`)
2. Reduce `max_length` parameter
3. Consider upgrading to a paid tier for GPU support

### Model Not Loading

If you see "Model not loaded" errors:

1. Check that `model/config.json` exists
2. Verify source files are in `src/aletheion/`
3. Check the app logs for import errors

## 🔄 Updating the Space

To update after changes:

```bash
cd hf_space

# Make your changes to app.py or other files

# Commit and push
git add .
git commit -m "Update: description of changes"
git push

# Space will automatically rebuild
```

## 📊 Performance Notes

### Free Tier Limitations

- **CPU only**: Inference will be slower than GPU
- **2GB RAM**: Limit sequence length and batch size
- **Timeout**: 60 seconds per request

### Recommended Settings for Free Tier

- Max sequence length: 100-200 tokens
- Temperature: 0.8-1.2
- Top-k: 40-50

### Upgrading to GPU

For faster inference:

1. Go to your Space settings
2. Select "Hardware" → "Upgrade"
3. Choose a GPU tier (T4 Small recommended)
4. Update `requirements.txt` to use GPU PyTorch:
   ```txt
   torch>=2.0.0,<3.0.0  # Remove CPU-only line
   ```

## 🔗 Useful Links

- **HuggingFace Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Documentation**: https://www.gradio.app/docs
- **Git LFS**: https://git-lfs.github.com/
- **Aletheion GitHub**: https://github.com/AletheionAGI/aletheion-llm

## 📧 Support

- **Issues**: https://github.com/AletheionAGI/aletheion-llm/issues
- **Email**: contact@alethea.tech
- **Discord**: .lacivo

## 🎉 Success Checklist

- [ ] HuggingFace account created
- [ ] Git LFS installed and initialized
- [ ] Space created on HuggingFace
- [ ] Repository cloned/initialized
- [ ] Files copied using `copy_to_hf.sh`
- [ ] Remote added to HuggingFace Space
- [ ] Files committed and pushed
- [ ] Space built successfully
- [ ] Demo is accessible and working
- [ ] (Optional) Model weights added
- [ ] Space card (README.md) looks good

Congratulations! Your Aletheion demo is now live on HuggingFace! 🎉
