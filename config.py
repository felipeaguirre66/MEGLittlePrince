from pathlib import Path
import os

# The Radeon RX 6600 XT (gfx1032) needs this ROCm compatibility override for
# GPU kernels in the current ROCm PyTorch build. Set it before importing torch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
from dataclasses import dataclass

DATA_DIR = Path('/mnt/sda/data-output/phd/Pallier2025')
READ_DIR = DATA_DIR / 'read' / 'ds007524-download'
LISTEN_DIR = DATA_DIR / 'listen' / 'ds007523-download'
RESULTS_DIR = Path('/mnt/sda/data-output/phd/Pallier2025_results')
WHISPER_TRANSCRIPTS_DIR = RESULTS_DIR / 'whisper_transcripts'
WAV_INPUT_DIR = LISTEN_DIR / 'sourcedata' / 'stimuli' / 'audio'


@dataclass
class LoggingConfig:
    """
    Configuration for logging.
    """
    level: str = "INFO"
    format: str = "[%(asctime)s][%(levelname)s] %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    obfuscate: bool = False  # Hide feature names (SR, WhC) in logs

def get_device() -> str:
    """Return a PyTorch device, including for AMD GPUs via ROCm."""
    # PyTorch exposes ROCm/HIP GPUs through its CUDA-compatible API, so the
    # correct device string is still "cuda" for both NVIDIA and AMD GPUs.
    return "cuda" if torch.cuda.is_available() else "cpu"