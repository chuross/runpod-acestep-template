import runpod
import os
import base64
import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Global handlers
dit_handler = None
llm_handler = None

def init_model():
    """Initializes the models once when the container starts."""
    global dit_handler, llm_handler
    print("Loading ACE-Step 1.5 Pipeline...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize handlers
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # Paths configuration
    # Assuming the code is running from the repo root /workspace
    project_root = "/workspace"
    checkpoint_dir = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/workspace/checkpoints")

    # Initialize services
    # config_path and lm_model_path might need to be adjusted based on the actual downloaded models
    # Assuming defaults or environment variables will be used by the library if not fully specified, 
    # but based on docs we need to specify them.
    # We will use sensible defaults for the template, user might need to change them.
    dit_handler.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-turbo", # or acestep-v15-full depending on what user wants/downloads
        device=device
    )

    llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path="acestep-5Hz-lm-0.6B", # or 1.7B
        backend="vllm", # Assumes vllm is installed and available
        device=device
    )
    
    print("Models loaded successfully.")

def handler(job):
    """Handler function executed for each request."""
    job_input = job["input"]
    
    # Extract parameters for GenerationParams
    prompt = job_input.get("prompt", "")
    lyrics = job_input.get("lyrics", "")
    duration = job_input.get("duration", 30)
    
    # Other potential params mapped from input
    bpm = job_input.get("bpm")
    key = job_input.get("key", "")
    vocal_language = job_input.get("vocal_language", "unknown")
    
    # Config parameters
    batch_size = job_input.get("batch_size", 1)
    
    # Validation
    if duration > 600: duration = 600
    
    try:
        # Configure generation parameters
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            bpm=bpm,
            keyscale=key,
            vocal_language=vocal_language,
            thinking=True # Enable thinking process by default
        )
        
        config = GenerationConfig(
            batch_size=batch_size,
            audio_format="mp3" # Request mp3 directly if supported, else convert
        )
        
        # Output directory
        save_dir = "/tmp/acestep_output"
        os.makedirs(save_dir, exist_ok=True)

        # Generate music
        result = generate_music(
            dit_handler=dit_handler, 
            llm_handler=llm_handler, 
            params=params, 
            config=config, 
            save_dir=save_dir
        )

        output_data = []
        
        if result.success:
            for audio in result.audios:
                file_path = audio['path']
                
                if file_path and os.path.exists(file_path):
                    with open(file_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                        
                    output_data.append({
                        "audio_base64": audio_base64,
                        "format": "mp3", # actual format depends on file extension
                        "metadata": {
                            "key": audio.get('key'),
                            "bpm": audio.get('bpm'),
                            "seed": audio.get('params', {}).get('seed')
                        }
                    })
                    
                    # Cleanup
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
            return output_data
        else:
            return {"error": str(result.error)}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        init_model()
    except Exception as e:
        print(f"FATAL: Failed to initialize models: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    print("Starting RunPod serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
