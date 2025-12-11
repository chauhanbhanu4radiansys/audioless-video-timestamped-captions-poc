# Quick Start Guide - BLIP-2 and MiniCPM-V

## 🚀 Fast Setup (5 minutes)

### Step 1: Install PyTorch

**For GPU (CUDA 12.x):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Core Dependencies

```bash
# IMPORTANT: torchvision is required for BLIP-2 fast processor
pip install torchvision transformers huggingface_hub pillow requests sentencepiece protobuf
```

### Step 3: Login to Hugging Face (Recommended)

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

### Step 4: Test Installation

**Test BLIP-2:**
```bash
python blip2_image_captioning.py --image-path test_image.jpg
```

**Test MiniCPM-V:**
```bash
python minicpmv_image_captioning.py --image-path test_image.jpg
```

---

## 📋 Prerequisites Checklist

### System Requirements
- ✅ Python 3.8+ (3.10+ recommended)
- ✅ 8GB+ RAM (16GB+ for MiniCPM-V)
- ✅ 15GB+ free disk space (for model downloads)
- ✅ Internet connection (for first-time model download)

### GPU Setup (Optional)
- ✅ NVIDIA GPU with CUDA support
- ✅ CUDA 11.8+ or 12.0+ installed
- ✅ 4GB+ VRAM (8GB+ for MiniCPM-V)

### Python Packages
- ✅ PyTorch (with CUDA if using GPU)
- ✅ transformers
- ✅ huggingface_hub
- ✅ pillow
- ✅ requests
- ✅ sentencepiece (for MiniCPM-V)
- ✅ protobuf (for MiniCPM-V)

---

## 🔧 Installation Commands

### Complete Installation (Copy & Paste)

```bash
# 1. Upgrade pip
pip install --upgrade pip

# 2. Install PyTorch (GPU version - adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install transformers and dependencies (torchvision is REQUIRED)
pip install torchvision transformers>=4.35.2 huggingface_hub>=0.19.4 pillow requests sentencepiece protobuf

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import BlipProcessor, Blip2ForConditionalGeneration; print('BLIP-2: OK')"
python -c "from transformers import AutoModel, AutoTokenizer; print('MiniCPM-V: OK')"
```

---

## 📝 Model-Specific Notes

### BLIP-2 (blip2_image_captioning.py)

**Model:** `Salesforce/blip2-flan-t5-xl` (Note: base variant doesn't exist, using XL)  
**Size:** ~5GB  
**Memory:** 6GB+ VRAM (GPU) or 12GB+ RAM (CPU)  
**Speed:** Fast (0.2-0.5s/image on GPU)

**Required:**
- `torchvision` package: `pip install torchvision`
- Model auto-downloads on first use

### MiniCPM-V (minicpmv_image_captioning.py)

**Model:** `openbmb/MiniCPM-V-2_6`  
**Size:** ~5GB  
**Memory:** 8GB+ VRAM (GPU) or 16GB+ RAM (CPU)  
**Speed:** Moderate (0.5-2s/image on GPU)

**Required:**
- Hugging Face login (recommended): `huggingface-cli login`
- `trust_remote_code=True` (handled automatically)
- Smaller batch sizes (default: 4)

---

## 🎯 First Run

### 1. Prepare Test Images

```bash
mkdir -p images
# Add some test images to the images/ directory
```

### 2. Run BLIP-2

```bash
python blip2_image_captioning.py images/ --batch-size 8
```

### 3. Run MiniCPM-V

```bash
python minicpmv_image_captioning.py images/ --batch-size 4
```

---

## ⚠️ Common Issues & Solutions

### "CUDA out of memory"
→ Reduce batch size: `--batch-size 2`

### "Model download failed"
→ Login to Hugging Face: `huggingface-cli login`

### "trust_remote_code error"
→ Update transformers: `pip install --upgrade transformers`

### "ModuleNotFoundError" or "torchvision not found"
→ Install torchvision: `pip install torchvision`

### "Repository Not Found" error (BLIP-2)
→ The model name has been updated to `Salesforce/blip2-flan-t5-xl` (base variant doesn't exist)

### Very slow on CPU
→ Normal! Use GPU if available, or reduce batch size

---

## 📊 Expected Performance

| Model | GPU Time/Image | CPU Time/Image | Recommended Batch Size |
|-------|----------------|----------------|------------------------|
| BLIP-2 | 0.1-0.3s | 2-5s | 8-32 (GPU), 4-8 (CPU) |
| MiniCPM-V | 0.5-2s | 5-15s | 4-8 (GPU), 2-4 (CPU) |

---

## 🔗 Useful Links

- **Full Setup Guide**: See `SETUP_GUIDE.md`
- **BLIP-2 Model**: https://huggingface.co/Salesforce/blip2-flan-t5-base
- **MiniCPM-V Model**: https://huggingface.co/openbmb/MiniCPM-V-2_6
- **Hugging Face Tokens**: https://huggingface.co/settings/tokens
- **PyTorch Installation**: https://pytorch.org/get-started/locally/

---

## 💡 Tips

1. **First run downloads models** (~5-10GB total) - be patient!
2. **GPU is 10-50x faster** - use it if available
3. **Start with small batches** and increase gradually
4. **Monitor memory** - use `nvidia-smi` for GPU, `htop` for CPU
5. **Save results** - use `--output results.json` to save captions
