"""
Run Whisper transcription (model sizes "base" and "large-v2") over every WAV
file in WAV_INPUT_DIR, saving each (audio, model_size) result as its own JSON
file in WHISPER_TRANSCRIPTS_DIR as soon as it is computed.
"""
import argparse
import logging
from datetime import date
from pathlib import Path

from config import WAV_INPUT_DIR, WHISPER_TRANSCRIPTS_DIR, get_device
from items import WhisperTranscript
from utils import get_logger
from whisper_transcripts import get_transcript

MODEL_SIZES = ["base", "large-v2"]
LOGS_DIR = Path(__file__).resolve().parent / "logs"

logger = get_logger()


def setup_file_logging() -> None:
    """Attach a file handler so logs are also written to logs/[date] run_pallier_whisper_transcripts.log."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"[{date.today().isoformat()}] run_pallier_whisper_transcripts.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)


def run(overwrite: bool = False) -> None:
    device = get_device()
    WHISPER_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(p for p in WAV_INPUT_DIR.iterdir() if p.suffix.lower() == ".wav")
    logger.info(f"Found {len(wav_paths)} WAV files in {WAV_INPUT_DIR}")

    # Loop model_size on the outside so each model is loaded once and reused across
    # all WAV files, instead of being swapped in/out of GPU memory for every file.
    for model_size in MODEL_SIZES:
        for wav_path in wav_paths:
            out_path = WhisperTranscript.path_for(wav_path, model_size)

            if out_path.exists() and not overwrite:
                logger.info(f"Skipping {wav_path.name} ({model_size}): {out_path.name} already exists")
                continue

            logger.info(f"Transcribing {wav_path.name} with model_size={model_size} (device={device})...")
            transcript = get_transcript(
                audio=str(wav_path),
                model_size=model_size,
                device=device,
                language="fr",
            )
            out_path.write_text(transcript.model_dump_json())
            logger.info(f"Saved transcript to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run transcription even if the output file already exists (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_file_logging()
    run(overwrite=args.overwrite)
