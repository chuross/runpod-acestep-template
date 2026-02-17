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
    project_root = "/workspace"
    checkpoint_dir = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/workspace/checkpoints")

    # ============================================================
    # OOM対策: vLLMのgpu_memory_utilizationを制限する
    # vLLMはデフォルトでVRAMの最大90%を確保しようとするため、
    # DiTモデルが載る余地がなくなりOOMになる。
    # max_ratioを0.5に制限して、DiTと共存できるようにする。
    # ============================================================
    LM_MAX_GPU_RATIO = float(os.environ.get("ACESTEP_LM_GPU_RATIO", "0.5"))
    _original_get_gpu_memory_utilization = llm_handler.get_gpu_memory_utilization

    def _patched_get_gpu_memory_utilization(model_path=None, minimal_gpu=3, min_ratio=0.1, max_ratio=0.9):
        ratio, low_gpu = _original_get_gpu_memory_utilization(
            model_path=model_path,
            minimal_gpu=minimal_gpu,
            min_ratio=min_ratio,
            max_ratio=LM_MAX_GPU_RATIO,  # max_ratioを制限
        )
        # 念のためratioも上限でクランプ
        ratio = min(ratio, LM_MAX_GPU_RATIO)
        print(f"[OOM対策] vLLM gpu_memory_utilization: {ratio:.3f} (max_ratio: {LM_MAX_GPU_RATIO})")
        return ratio, low_gpu

    llm_handler.get_gpu_memory_utilization = _patched_get_gpu_memory_utilization

    # ============================================================
    # 初期化順序: LLM → DiT (vLLMに先にメモリを確保させる)
    # ============================================================
    print("Initializing LLM handler (vLLM)...")
    llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-4B"),
        backend="vllm",
        device=device
    )
    if not llm_handler.llm_initialized:
        raise RuntimeError("Failed to initialize LLM handler")
    print("LLM handler initialized.")

    print("Initializing DiT model...")
    status, ok = dit_handler.initialize_service(
        project_root=project_root,
        config_path=os.environ.get("ACESTEP_MODEL_CONFIG", "acestep-v15-turbo"), 
        device=device
    )
    if not ok:
        raise RuntimeError(f"Failed to initialize DiT model: {status}")
    print(f"DiT model initialized: {status}")
    
    print("Models loaded successfully.")

def handler(job):
    """Handler function executed for each request."""
    job_input = job["input"]
    
    # Extract parameters for GenerationParams
    prompt = job_input.get("prompt", "")
    lyrics = job_input.get("lyrics", "")
    duration = job_input.get("duration", -1)
    
    # Other potential params mapped from input
    bpm = job_input.get("bpm")
    key = job_input.get("key", "")
    vocal_language = job_input.get("vocal_language", "unknown")
    
    # Config parameters
    batch_size = job_input.get("batch_size", 1)
    
    # High quality generation parameters (overridable from request)
    # High quality generation parameters (overridable from request)
    # Default values optimized for ACE-Step-1.5-Turbo
    inference_steps = job_input.get("inference_steps", 8)  # Turbo: 8 steps is standard
    guidance_scale = job_input.get("guidance_scale", 0.0)  # Turbo: 0.0 (distilled) or low CFG
    use_adg = job_input.get("use_adg", True)
    shift = job_input.get("shift", 3.0)  # Turbo: 3.0 recommended
    seed = job_input.get("seed", -1)
    thinking = job_input.get("thinking", True)
    
    # Advanced DiT parameters
    infer_method = job_input.get("infer_method", "ode") # "ode" (Euler) or "sde"
    
    # LM parameters
    lm_temperature = job_input.get("lm_temperature", 0.85)
    lm_cfg_scale = job_input.get("lm_cfg_scale", 2.0)
    audio_format = job_input.get("audio_format", "mp3")
    
    # Validation
    if duration > 600: duration = 600
    
    # Check for instrumental request
    instrumental = "[Instrumental]" in lyrics if lyrics else False
    
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
            instrumental=instrumental, # instrumental flag based on lyrics content
            thinking=thinking,
            # High Quality Settings (Base Model)
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            use_adg=use_adg,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=shift,
            infer_method=infer_method,
            seed=seed,
            # LM parameters
            lm_temperature=lm_temperature,
            lm_cfg_scale=lm_cfg_scale,
        )
        
        config = GenerationConfig(
            batch_size=batch_size,
            audio_format=audio_format,
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
                        "format": audio_format,
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
