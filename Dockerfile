# Base image: PyTorch with CUDA support
FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Fix model checkpoint directory for caching
ENV ACESTEP_CHECKPOINT_DIR=/workspace/checkpoints

WORKDIR /workspace

# Install system dependencies
# ffmpeg is required for audio processing
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install RunPod SDK
RUN pip install runpod

# Clone ACE-Step 1.5 repository
RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git .

# Install dependencies
RUN uv sync

# Pre-download models to speed up cold starts
# This will increase image size but reduce startup time significantly
RUN uv run acestep-download

# Copy handler code
COPY handler.py /workspace/handler.py

# Command to run the handler
CMD ["python", "-u", "/workspace/handler.py"]
