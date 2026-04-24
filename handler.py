import os
import runpod
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import requests
import tempfile
import gc

# Environment setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Global model cache to avoid reloading for every job
print(f"[*] Initializing WhisperX with default model={DEFAULT_MODEL_SIZE}, device={DEVICE}, compute_type={COMPUTE_TYPE}")
# Use a dictionary to potentially support multiple models, though memory is a limit
models = {
    DEFAULT_MODEL_SIZE: whisperx.load_model(DEFAULT_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
}

def handler(job):
    """
    RunPod handler function.
    Expected input: {
        "url": "https://example.com/audio.mp3",
        "min_speakers": 1,
        "max_speakers": 5,
        "language": "en",
        "model": "large-v3"
    }
    """
    job_input = job["input"]
    audio_url = job_input.get("url")
    hf_token = os.getenv("HF_TOKEN")
    
    # Extract parameters from input
    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")
    language = job_input.get("language")
    requested_model = job_input.get("model", DEFAULT_MODEL_SIZE)
    
    if not audio_url:
        return {"error": "Missing 'url' in input"}
    
    if not hf_token:
        return {"error": "HF_TOKEN environment variable is missing. Diarization requires it."}

    # Load/Get model
    if requested_model not in models:
        print(f"[*] Loading requested model: {requested_model}")
        try:
            models[requested_model] = whisperx.load_model(requested_model, DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            return {"error": f"Failed to load model '{requested_model}': {str(e)}"}
    
    current_model = models[requested_model]

    # 1. Download audio to a temporary file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        print(f"[*] Downloading audio from: {audio_url}")
        response = requests.get(audio_url, stream=True, timeout=60)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            temp_audio.write(chunk)
        temp_audio.close()
        audio_path = temp_audio.name
    except Exception as e:
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)
        return {"error": f"Failed to download audio: {str(e)}"}

    try:
        # 2. Transcribe
        print(f"[*] Transcribing (Model: {requested_model})...")
        audio = whisperx.load_audio(audio_path)
        
        transcribe_args = {"batch_size": BATCH_SIZE}
        if language:
            transcribe_args["language"] = language
            
        result = current_model.transcribe(audio, **transcribe_args)
        
        # 3. Align (get precise word-level timestamps)
        detected_language = result.get("language", language)
        print(f"[*] Aligning (Language: {detected_language})...")
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        # Free up align model memory after use
        del model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 4. Diarization
        print("[*] Performing Speaker Diarization...")
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
        
        diarize_args = {}
        if min_speakers is not None:
            diarize_args["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            diarize_args["max_speakers"] = int(max_speakers)
            
        diarize_segments = diarize_model(audio, **diarize_args)
        
        # Free up diarize model memory
        del diarize_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 5. Assign speakers to words
        print("[*] Assigning speakers to word-level segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Cleanup local file
        os.remove(audio_path)
        
        print("[*] Processing complete.")
        return result

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        import traceback
        traceback.print_exc()
        print(f"[!] Error: {str(e)}")
        return {"error": f"Internal processing error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
