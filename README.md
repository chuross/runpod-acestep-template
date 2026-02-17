# AceStep 1.5 on RunPod Serverless

This repository contains a template for deploying AceStep 1.5 (an open-source music generation model) on RunPod Serverless.

## Requirements

- RunPod account
- Docker installed locally (or use RunPod's image builder)
- Container registry (Docker Hub, GitHub Container Registry, etc.)

## Deployment Steps

### 1. Build the Docker Image

```bash
docker build -t your-username/acestep-runpod:v1 .
```

> **Note:** The build process includes downloading model weights (approx. 15GB), so it may take some time.

### 2. Push to Registry

```bash
docker push your-username/acestep-runpod:v1
```

### 3. Configure RunPod

1. Go to RunPod Console > Serverless > My Templates > New Template.
2. Fill in the details:
    - **Template Name**: `AceStep-1.5`
    - **Container Image**: `your-username/acestep-runpod:v1`
    - **Container Disk**: `20 GB` (or more, recommended)
    - **Container RAM**: `16 GB` (or more)
    - **Docker Command**: (Leave empty, `CMD` from Dockerfile will be used) or strictly `python -u /workspace/handler.py`
3. Create the template.
4. Go to Serverless > New Endpoint.
5. Select the template you just created.
6. Choose a GPU. **Recommended: 24GB VRAM or higher (e.g., A10G, A100, RTX 3090, RTX 4090)** due to the large model size.

## Usage

Send a request to your endpoint:

```json
{
  "input": {
    "prompt": "Epic orchestral battle music",
    "lyrics": "",
    "duration": 30,
    "batch_size": 1
  }
}
```

### Parameters

- `prompt`: Description of the music to generate.
- `lyrics`: Lyrics for the song (optional).
- `duration`: Duration in seconds (max 600 recommended).
- `batch_size`: Number of variations to generate.

## Response

The response will contain a list of generated audio files encoded in Base64.

```json
[
  {
    "audio_base64": "...",
    "format": "mp3",
    "metadata": {}
  }
]
```
