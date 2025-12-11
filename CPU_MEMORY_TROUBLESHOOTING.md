# CPU Memory Troubleshooting Guide

## Issue: Process Killed During Model Loading

If you see `[1] 71027 killed` when loading MiniCPM-V on CPU, this indicates **Out of Memory (OOM)**.

### Root Cause

MiniCPM-V 2.6 is a **2.4B parameter model** that requires significant RAM:
- **Minimum**: ~12-16 GB available RAM
- **Recommended**: 16-32 GB RAM for stable operation

The model gets killed by the OS when it runs out of memory during loading.

---

## Solutions

### Solution 1: Check Available Memory

First, check how much RAM you have available:

```bash
# On macOS/Linux
free -h
# or
vm_stat  # macOS only

# Check available memory in Python
python -c "import psutil; mem = psutil.virtual_memory(); print(f'Available: {mem.available/(1024**3):.1f} GB / Total: {mem.total/(1024**3):.1f} GB')"
```

### Solution 2: Free Up Memory

Before running the script:
1. **Close other applications** (browsers, IDEs, etc.)
2. **Restart your terminal/computer** to clear cached memory
3. **Stop other Python processes**: `pkill -f python`
4. **Check running processes**: `top` or `Activity Monitor` (macOS)

### Solution 3: Use Memory-Optimized Loading

The script now includes automatic memory optimizations:
- ✅ `low_cpu_mem_usage=True` - Reduces peak memory during loading
- ✅ `device_map="cpu"` - Efficient CPU memory mapping
- ✅ Float16 loading attempt (50% memory reduction)
- ✅ Memory check before loading (if `psutil` installed)

**Install psutil for memory checking:**
```bash
pip install psutil
```

### Solution 4: Use BLIP-2 Instead (Lower Memory)

BLIP-2 requires **~8GB RAM** (half of MiniCPM-V):

```bash
python blip2_image_captioning.py image.jpg
```

**Trade-offs:**
- ✅ Uses less memory (~8GB vs ~16GB)
- ✅ Faster on CPU (~2-5s vs ~5-15s per image)
- ⚠️ Slightly lower quality captions (but still very good)

### Solution 5: Use GPU Instead

If you have a GPU available, MiniCPM-V will use much less system RAM:

```bash
# Check if GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If GPU available, the script will automatically use it
python minicpmv_image_captioning.py image.jpg
```

**GPU Requirements:**
- ~8GB VRAM (vs 16GB system RAM)
- Much faster inference (~0.5-2s vs ~5-15s)

### Solution 6: Increase Swap Space (macOS/Linux)

**⚠️ Warning**: Swap is slower than RAM, but can help prevent OOM kills.

**macOS:**
```bash
# Check current swap
sysctl vm.swapusage

# macOS manages swap automatically, but you can:
# 1. Close applications to free RAM
# 2. Restart to clear memory
```

**Linux:**
```bash
# Check swap
free -h

# Create swap file (if needed)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Memory Requirements Summary

| Model | CPU RAM | GPU VRAM | Disk Space |
|-------|---------|----------|------------|
| **BLIP-2** | 8GB+ | 4GB+ | ~5GB |
| **MiniCPM-V 2.6** | **16GB+** | 8GB+ | ~10GB |

---

## Quick Diagnostic Commands

```bash
# 1. Check available memory
python -c "import psutil; m=psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.1f}GB / Total: {m.total/(1024**3):.1f}GB')"

# 2. Check if GPU available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Test with BLIP-2 (lower memory)
python blip2_image_captioning.py image.jpg

# 4. Monitor memory during loading
# In another terminal:
watch -n 1 'free -h'  # Linux
# or Activity Monitor on macOS
```

---

## Recommended Approach

1. **If you have < 16GB RAM**: Use **BLIP-2** instead
   ```bash
   python blip2_image_captioning.py image.jpg
   ```

2. **If you have 16GB+ RAM**: 
   - Close other applications
   - Install `psutil`: `pip install psutil`
   - Try MiniCPM-V again

3. **If you have GPU**: Use GPU mode (automatic if CUDA available)

---

## Still Having Issues?

1. **Check system logs** for OOM kills:
   ```bash
   # macOS
   log show --predicate 'eventMessage contains "killed"' --last 1h
   
   # Linux
   dmesg | grep -i "killed process"
   ```

2. **Try with a smaller test image** first

3. **Use BLIP-2** as a reliable alternative

4. **Consider using cloud GPU** (RunPod, Google Colab, etc.) if local resources are limited

---

## Performance Expectations

### BLIP-2 (Lower Memory Option)
- **CPU**: ~2-5 seconds per image
- **GPU**: ~0.1-0.3 seconds per image
- **Memory**: ~8GB RAM

### MiniCPM-V 2.6 (Higher Quality)
- **CPU**: ~5-15 seconds per image
- **GPU**: ~0.5-2 seconds per image
- **Memory**: ~16GB RAM

---

## Additional Resources

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Transformers Memory Optimization](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
- [MiniCPM-V Model Card](https://huggingface.co/openbmb/MiniCPM-V-2_6)
