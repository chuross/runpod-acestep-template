# Base image: PyTorch with CUDA support
FROM runpod/pytorch:1.0.3-cu1300-torch260-ubuntu2404

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ACESTEP_CHECKPOINT_DIR=/workspace/checkpoints
# Disable tqdm progress bars in non-interactive mode
ENV ACESTEP_DISABLE_TQDM=1

WORKDIR /workspace

# Install system dependencies
# ffmpeg is required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone ACE-Step 1.5 repository

# Clone ACE-Step 1.5 repository into temporary directory and copy to workspace
RUN git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git /tmp/acestep && \
    cp -rf /tmp/acestep/. . && \
    rm -rf /tmp/acestep

# Install dependencies
RUN uv sync --no-dev

# Install RunPod SDK
RUN uv add runpod

# 2. SFT DiT model
RUN uv run acestep-download

# 3. 4B LM for best quality (requires ≥24GB VRAM)
RUN uv run acestep-download --model acestep-5Hz-lm-4B

# Copy handler code
COPY handler.py /workspace/handler.py

# Command to run the handler via uv (ensures venv with all deps is used)
CMD ["uv", "run", "python", "-u", "/workspace/handler.py"]
