from pathlib import Path
from typing import Dict, Any, Union
from pydantic import BaseModel
import pandas as pd # type: ignore

import config

class WhisperTranscript(BaseModel):
    whisper_result: Dict[str, Any]
    model_size: str
    language: Union[str, None]
    params: Dict[str, Any]

    @staticmethod
    def path_for(wav_path: Union[str, Path], model_size: str) -> Path:
        """Path of the JSON file for a given WAV file and model size, as saved by run_pallier_whisper_transcripts.py."""
        return config.WHISPER_TRANSCRIPTS_DIR / f"{Path(wav_path).stem}_whisper_{model_size}.json"

    @classmethod
    def from_wav_path(cls, wav_path: Union[str, Path], model_size: str) -> "WhisperTranscript":
        """Load the saved transcript for a given WAV file and model size."""
        return cls.model_validate_json(cls.path_for(wav_path, model_size).read_text())