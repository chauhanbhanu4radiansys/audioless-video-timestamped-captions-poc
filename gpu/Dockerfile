FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install Python and system dependencies
RUN python3.11 -m pip install --upgrade pip && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install Python dependencies
COPY ./gpu/requirements.txt /requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /requirements.txt

# Create model cache directories
RUN mkdir -p /model_cache/huggingface && \
    mkdir -p /model_cache/transformers

# Set cache environment variables
ENV HF_HOME=/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/model_cache/transformers
ENV HF_HUB_CACHE=/model_cache/huggingface

# Pre-download BLIP model during build (CRITICAL - this is what you need!)
RUN echo "Pre-downloading BLIP captioning model..." && \
    python3.11 << 'PYEOF'
import os
os.environ['HF_HOME'] = '/model_cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/model_cache/transformers'
os.environ['HF_HUB_CACHE'] = '/model_cache/huggingface'

from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import snapshot_download

print('Downloading BLIP model files...')
snapshot_download('Salesforce/blip-image-captioning-large',
                  cache_dir='/model_cache/huggingface',
                  local_files_only=False)

print('Loading BLIP processor...')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large',
                                          cache_dir='/model_cache/huggingface')

print('Loading BLIP model...')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large',
                                                      cache_dir='/model_cache/huggingface')

print('✓ BLIP model downloaded and cached')
PYEOF

# Copy application files
COPY ./src /src
COPY ./gpu/handler.py /handler.py
COPY ./gpu/local.py /local.py
COPY ./main.py /main.py

# Set environment variables
ENV PYTHONPATH=/
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3.11", "-u", "/handler.py"]
