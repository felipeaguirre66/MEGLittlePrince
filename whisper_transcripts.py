from typing import Optional
from items import WhisperTranscript
import gc
import inspect
import torch
import whisper_timestamped # type: ignore
import config
from utils import get_logger
logger = get_logger()

_MODEL_CACHE = {}

def _unload_other_models(keep_model_size: str) -> None:
    """Evict every cached model other than keep_model_size, freeing its (GPU) memory."""
    for model_size in list(_MODEL_CACHE):
        if model_size != keep_model_size:
            logger.info(f"Unloading Whisper model ({model_size}) to free memory")
            del _MODEL_CACHE[model_size]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_size: str = "large-v2", device: str = "cpu"):
    # Keep at most one model in memory at a time (large-v2 + base together can exceed GPU memory).
    if model_size not in _MODEL_CACHE:
        _unload_other_models(keep_model_size=model_size)
        logger.info(f"Loading Whisper model ({model_size}, device: {device})...")
        _MODEL_CACHE[model_size] = whisper_timestamped.load_model(model_size, device=device)
    else:
        logger.info(f"Cache-loading Whisper model ({model_size}, device: {device})")
    return _MODEL_CACHE[model_size]

def load_all_whisper_models():
    """Load each model size once; since only one is kept cached at a time, this just warms each in turn."""
    model_sizes = ["base", "large-v2"]
    for model_size in model_sizes:
        get_model(model_size=model_size, device=config.get_device())

def get_transcript(
    audio: str,
    model_size: str = "large-v2",
    device: str = "cpu",
    initial_prompt: Optional[str] = None,
    fp16: bool = False,
    language: Optional[str] = "fr",
    do_vad: Optional[bool] = False,
) -> WhisperTranscript:
    """
    Get a whisper transcription from a WAV file path.
    """
    model = get_model(model_size, device=device)

    audio_data = whisper_timestamped.load_audio(audio)
    result = whisper_timestamped.transcribe(
        model, audio_data, language=language, initial_prompt=initial_prompt, fp16=fp16, vad=do_vad,
    )

    # Collect args and results
    signature = inspect.signature(whisper_timestamped.transcribe)
    default_args = {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not inspect.Parameter.empty
    }
    actual_args = {
        "language": language,
        "initial_prompt": initial_prompt,
        "fp16": fp16,
        "vad": do_vad,
    }
    params = {**default_args, **actual_args}

    return WhisperTranscript(
        whisper_result=result,
        model_size=model_size,
        language=language,
        params=params
    )