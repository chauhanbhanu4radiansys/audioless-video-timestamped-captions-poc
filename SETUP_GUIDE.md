# Setup Guide for BLIP-2 and MiniCPM-V Image Captioning

This guide covers the prerequisite steps to use both `blip2_image_captioning.py` and `minicpmv_image_captioning.py`.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Install Dependencies](#install-dependencies)
4. [Model-Specific Setup](#model-specific-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 
  - BLIP-2: 8GB minimum (16GB+ recommended)
  - MiniCPM-V: 16GB minimum (32GB+ recommended)
- **Disk Space**: 
  - BLIP-2: ~5GB for model files
  - MiniCPM-V: ~10GB for model files

### GPU Requirements (Optional but Recommended)
- **CUDA**: 11.8+ or 12.0+ (for GPU acceleration)
- **GPU Memory**:
  - BLIP-2: 4GB+ VRAM
  - MiniCPM-V: 8GB+ VRAM
- **GPU**: NVIDIA GPU with CUDA support

### CPU-Only Setup
Both models can run on CPU, but will be significantly slower:
- BLIP-2: ~2-5 seconds per image on CPU
- MiniCPM-V: ~5-15 seconds per image on CPU

---

## Python Environment Setup

### 1. Create a Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

---

## Install Dependencies

### Core Dependencies

Install the base packages required for both models:

```bash
# PyTorch with torchvision (torchvision is REQUIRED for fast image processors)
pip install torch torchvision torchaudio
pip install transformers
pip install pillow
pip install requests
```

**Important:** `torchvision` is required for the fast image processor in BLIP-2. Without it, the script will fall back to the slow processor.

### PyTorch Installation (GPU Support)

**For CUDA 12.x (Recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Model-Specific Setup

### BLIP-2 (Flan-T5-base) Setup

#### 1. Install Additional Dependencies

```bash
# BLIP-2 uses standard transformers library
# torchvision is REQUIRED for fast image processor
pip install torchvision
```

#### 2. Model Download

The model will be automatically downloaded on first use from Hugging Face:
- Model: `Salesforce/blip2-flan-t5-xl` (Note: blip2-flan-t5-base doesn't exist, using XL variant)
- Size: ~5GB
- First run will download automatically
- **Note**: This is the Flan-T5-XL variant (not base), which is the smallest Flan-T5 variant available

#### 3. Hugging Face Authentication (Optional)

If you encounter rate limits, create a Hugging Face account and login:

```bash
pip install huggingface_hub
huggingface-cli login
```

#### 4. Test BLIP-2 Installation

```bash
python blip2_image_captioning.py --image-path test_image.jpg
```

---

### MiniCPM-V 2.6 Setup

#### 1. Install Additional Dependencies

MiniCPM-V requires `trust_remote_code=True`, which means it may need additional dependencies:

```bash
# Ensure you have the latest transformers
pip install --upgrade transformers

# May require additional packages (installed automatically on first run)
# Common ones include:
pip install sentencepiece  # Often needed for tokenizers
pip install protobuf       # Sometimes required
```

#### 2. Model Download

The model will be automatically downloaded on first use:
- Model: `openbmb/MiniCPM-V-2_6`
- Size: ~5GB
- First run will download automatically
- **Note**: Requires `trust_remote_code=True` (handled automatically in the script)

#### 3. Hugging Face Authentication (Recommended)

MiniCPM-V models may require Hugging Face authentication:

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your Hugging Face token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

#### 4. GPU Memory Optimization

MiniCPM-V is memory-intensive. If you encounter OOM errors:

```bash
# Use smaller batch sizes
python minicpmv_image_captioning.py --batch-size 2

# Or reduce max_length
python minicpmv_image_captioning.py --max-length 50
```

#### 5. Test MiniCPM-V Installation

```bash
python minicpmv_image_captioning.py --image-path test_image.jpg
```

---

## Complete Installation Script

Create a file `setup_models.sh` (or `setup_models.bat` for Windows):

```bash
#!/bin/bash
# Setup script for BLIP-2 and MiniCPM-V

echo "Setting up Python environment for image captioning models..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install transformers pillow requests

# Install additional dependencies
echo "Installing additional dependencies..."
pip install sentencepiece protobuf huggingface_hub

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import BlipProcessor, Blip2ForConditionalGeneration; print('BLIP-2: OK')"
python -c "from transformers import AutoModel, AutoTokenizer; print('MiniCPM-V: OK')"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face (recommended): huggingface-cli login"
echo "2. Test BLIP-2: python blip2_image_captioning.py --image-path test.jpg"
echo "3. Test MiniCPM-V: python minicpmv_image_captioning.py --image-path test.jpg"
```

Make it executable:
```bash
chmod +x setup_models.sh
./setup_models.sh
```

---

## Verification

### 1. Check Python Version

```bash
python --version  # Should be 3.8+
```

### 2. Check PyTorch Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 3. Test BLIP-2 Import

```python
from transformers import BlipProcessor, Blip2ForConditionalGeneration
print("BLIP-2 imports successful!")
```

### 4. Test MiniCPM-V Import

```python
from transformers import AutoModel, AutoTokenizer
print("MiniCPM-V imports successful!")
```

### 5. Quick Test Run

**BLIP-2:**
```bash
# Create a test image directory
mkdir -p test_images
# Add some test images, then:
python blip2_image_captioning.py test_images --batch-size 4
```

**MiniCPM-V:**
```bash
python minicpmv_image_captioning.py test_images --batch-size 2
```

---

## Usage Examples

### BLIP-2 Usage

```bash
# Basic usage
python blip2_image_captioning.py images/

# With custom batch size
python blip2_image_captioning.py images/ --batch-size 16

# Conditional captioning
python blip2_image_captioning.py images/ --conditional --prompt "a photography of"

# Save results to JSON
python blip2_image_captioning.py images/ --output results_blip2.json
```

### MiniCPM-V Usage

```bash
# Basic usage (smaller batch size recommended)
python minicpmv_image_captioning.py images/ --batch-size 4

# With custom prompt
python minicpmv_image_captioning.py images/ --conditional --prompt "What objects are visible in this image?"

# Save results to JSON
python minicpmv_image_captioning.py images/ --output results_minicpmv.json
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""` before running
- Close other GPU applications

### Issue: "trust_remote_code" error (MiniCPM-V)

**Solution:**
- The script handles this automatically, but if you see errors:
- Ensure you have the latest transformers: `pip install --upgrade transformers`
- Login to Hugging Face: `huggingface-cli login`

### Issue: Model download fails

**Solutions:**
- Check internet connection
- Login to Hugging Face: `huggingface-cli login`
- Set Hugging Face cache directory:
  ```bash
  export HF_HOME=/path/to/cache
  ```

### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Solution:**
```bash
pip install transformers
```

### Issue: Slow performance on CPU

**Solutions:**
- Use GPU if available
- Reduce batch size
- Reduce max_length
- Consider using BLIP-2 instead of MiniCPM-V (BLIP-2 is faster on CPU)

### Issue: "RuntimeError: CUDA error" (GPU)

**Solutions:**
- Check CUDA installation: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Try CPU mode: `CUDA_VISIBLE_DEVICES="" python script.py`

---

## Memory Requirements Summary

| Model | CPU RAM | GPU VRAM | Disk Space |
|-------|---------|----------|------------|
| BLIP-2 | 8GB+ | 4GB+ | ~5GB |
| MiniCPM-V | 16GB+ | 8GB+ | ~10GB |

---

## Performance Expectations

### BLIP-2 (Flan-T5-base)
- **GPU**: ~0.1-0.3 seconds per image
- **CPU**: ~2-5 seconds per image
- **Batch size**: 8-32 (GPU), 4-8 (CPU)

### MiniCPM-V 2.6
- **GPU**: ~0.5-2 seconds per image
- **CPU**: ~5-15 seconds per image
- **Batch size**: 4-8 (GPU), 2-4 (CPU)

---

## Additional Resources

- **BLIP-2 Model Card**: https://huggingface.co/Salesforce/blip2-flan-t5-base
- **MiniCPM-V Model Card**: https://huggingface.co/openbmb/MiniCPM-V-2_6
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **PyTorch Installation**: https://pytorch.org/get-started/locally/

---

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (with CUDA if GPU available)
- [ ] Transformers library installed
- [ ] Pillow, requests installed
- [ ] Hugging Face CLI installed (optional but recommended)
- [ ] Hugging Face account created and logged in (for MiniCPM-V)
- [ ] Test images directory created
- [ ] BLIP-2 test run successful
- [ ] MiniCPM-V test run successful

---

## Notes

1. **First Run**: Both models will download large files on first use (~5-10GB total). Ensure stable internet connection.

2. **Hugging Face Login**: While not always required, logging in helps avoid rate limits and may be required for some models.

3. **GPU vs CPU**: GPU provides 10-50x speedup. If you have a GPU, ensure CUDA is properly installed.

4. **Batch Sizes**: Start with smaller batch sizes and increase gradually to find optimal settings for your hardware.

5. **Memory**: Monitor memory usage, especially with MiniCPM-V. Use `nvidia-smi` (GPU) or `htop` (CPU) to monitor.
