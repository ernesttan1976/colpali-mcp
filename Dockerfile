FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    wget \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# First ensure pip is up to date
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first for better caching
COPY requirements.txt .

# Create a modified requirements file without PyTorch packages
RUN grep -v "torch\|torchaudio\|torchvision" requirements.txt > requirements_filtered.txt

# Install PyTorch with CUDA 12.6 support
# Using version 2.7.0+cu126 which is available with CUDA 12.6
RUN pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
RUN pip install -r requirements_filtered.txt

# Create directories for storage and model files
RUN mkdir -p ./data/embeddings_db ./models/colqwen2/model ./models/colqwen2/processor

# Create config.json files in the container directly instead of copying them
RUN echo '{\n\
  "model_type": "colqwen2",\n\
  "architectures": ["ColQwen2ForVisionText"],\n\
  "_name_or_path": "vidore/colqwen2-v1.0",\n\
  "hidden_size": 2048,\n\
  "num_hidden_layers": 24,\n\
  "num_attention_heads": 16,\n\
  "intermediate_size": 8192,\n\
  "hidden_act": "silu",\n\
  "max_position_embeddings": 4096,\n\
  "use_cache": true,\n\
  "bos_token_id": 1,\n\
  "eos_token_id": 2,\n\
  "rope_theta": 10000.0,\n\
  "use_flash_attn": false,\n\
  "tie_word_embeddings": false,\n\
  "pad_token_id": 0,\n\
  "vocab_size": 151936,\n\
  "attention_dropout": 0.0,\n\
  "initializer_range": 0.02,\n\
  "layernorm_epsilon": 1e-5,\n\
  "rms_norm_eps": 1e-6,\n\
  "transformers_version": "4.47.1",\n\
  "torch_dtype": "bfloat16"\n\
}' > ./models/colqwen2/model/config.json

RUN echo '{\n\
  "processor_class": "ColQwen2Processor",\n\
  "is_vision_text_model": true,\n\
  "image_size": 224,\n\
  "image_mean": [0.48145466, 0.4578275, 0.40821073],\n\
  "image_std": [0.26862954, 0.26130258, 0.27577711],\n\
  "feature_extractor_type": "ColQwen2FeatureExtractor",\n\
  "tokenizer_class": "PreTrainedTokenizerFast",\n\
  "model_max_length": 4096,\n\
  "padding_side": "right",\n\
  "truncation_side": "right",\n\
  "pad_token_id": 0,\n\
  "eos_token_id": 2,\n\
  "bos_token_id": 1,\n\
  "sep_token_id": 3,\n\
  "pad_token": "<|endoftext|>",\n\
  "eos_token": "<|im_end|>",\n\
  "bos_token": "<|im_start|>",\n\
  "do_normalize": true,\n\
  "do_resize": true,\n\
  "do_center_crop": true,\n\
  "vision_text_processor": true,\n\
  "transformers_version": "4.47.1"\n\
}' > ./models/colqwen2/processor/config.json

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Copy verification script and startup script
COPY verify_cuda.py .
COPY start-app.sh .
COPY docker-entrypoint.sh .
RUN chmod +x start-app.sh docker-entrypoint.sh

# Don't try to install flash-attn at build time - it's too slow and often fails
# Instead, we'll let the app try to install it at runtime if needed
RUN apt-get update && \
    apt-get install -y ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Make sure we use our entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]

# Command to run the application
CMD ["./start-app.sh"]