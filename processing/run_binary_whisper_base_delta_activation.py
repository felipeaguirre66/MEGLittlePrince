"""Compute low-minus-high Whisper base confidence MEG activation for each subject/run.

Run from the repository root with
``python -m processing.run_binary_whisper_base_delta_activation --workers N``.
"""

from __future__ import annotations

import os

# Must be set before numpy/mne are imported: default multi-threaded BLAS is much slower for this workload.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING
import warnings

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LISTEN_DIR, RESULTS_DIR

if TYPE_CHECKING:
    from meg import Meg


BEFORE_START_MS = -500
AFTER_START_MS = 800
OUTPUT_DIR = RESULTS_DIR / 'binary_whisper_base_confidence'
EVENTS_GLOB = 'sub-*/ses-01/meg/*_task-listen_run-*_events.tsv'


def _median_activation_by_word(
    meg: Meg,
    word_indices: list[int],
    before_start_ms: float = BEFORE_START_MS,
    after_start_ms: float = AFTER_START_MS,
) -> np.ndarray:
    """Return the channel-by-sample median across words, extracting each word's epoch once."""
    if not word_indices:
        raise ValueError('Cannot compute median activation for an empty word group')

    start_offset = round(before_start_ms * meg.sfreq / 1000)
    stop_offset = round(after_start_ms * meg.sfreq / 1000)
    n_samples = stop_offset - start_offset
    if n_samples <= 0:
        raise ValueError(
            f'Invalid epoch window: {before_start_ms} ms to {after_start_ms} ms'
        )

    n_channels = len(meg.grad_picks)
    epochs = np.empty((len(word_indices), n_channels, n_samples), dtype=np.float32)
    for word_position, word_index in enumerate(word_indices):
        epochs[word_position] = meg.get_grad_by_word(
            word_index,
            before_start_ms=before_start_ms,
            after_start_ms=after_start_ms,
        )
    return np.median(epochs, axis=0)


def _process_run(events_path: Path) -> None:
    from meg import Meg

    output_path = _output_path_for_run(events_path)
    if output_path.is_file():
        return

    run_prefix = events_path.name.removesuffix('_events.tsv')

    meg = Meg(run_prefix, filter=True, maxwell=True, add_features=True)
    meg.raw_before_maxwell = None  # not needed past this point; frees memory for the epoch step
    if 'low_base_confidence' not in meg.events.columns:
        raise ValueError(f"'low_base_confidence' is missing from events for {run_prefix}")
    if meg.events['low_base_confidence'].isna().any():
        raise ValueError(f"'low_base_confidence' contains missing values for {run_prefix}")

    low_word_indices = meg.events.index[meg.events['low_base_confidence']].tolist()
    high_word_indices = meg.events.index[~meg.events['low_base_confidence']].tolist()
    if not low_word_indices or not high_word_indices:
        raise ValueError(
            f'Both confidence groups must contain words for {run_prefix}; '
            f'found {len(low_word_indices)} low and {len(high_word_indices)} high'
        )

    low_median = _median_activation_by_word(meg, low_word_indices)
    high_median = _median_activation_by_word(meg, high_word_indices)
    signal = low_median - high_median

    start_offset = round(BEFORE_START_MS * meg.sfreq / 1000)
    times = (np.arange(signal.shape[1]) + start_offset) / meg.sfreq
    channel_names = np.asarray([meg.raw.ch_names[pick] for pick in meg.grad_picks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.npz', dir=output_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                signal=signal,
                channel_names=channel_names,
                times=times,
                low_word_count=np.asarray(len(low_word_indices)),
                high_word_count=np.asarray(len(high_word_indices)),
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()



def _process_run_safely(events_path: Path) -> str | None:
    """Process one run. Returns an error message, or None on success."""
    warnings.simplefilter('ignore')
    try:
        _process_run(events_path)
    except Exception:
        return traceback.format_exc()
    return None


def _output_path_for_run(events_path: Path) -> Path:
    run_prefix = events_path.name.removesuffix('_events.tsv')
    match = re.fullmatch(
        r'(sub-\d+)_ses-01_task-listen_(run-\d+)',
        run_prefix,
    )
    if match is None:
        raise ValueError(f'Unexpected events filename: {events_path.name}')
    subject, _ = match.groups()
    output_dir = OUTPUT_DIR / subject / run_prefix / 'meg'
    return output_dir / f'{run_prefix}_low-minus-high_binary_whisper_base_confidence.npz'


def run_binary_whisper_base_delta_activation(workers: int = 1) -> None:
    """Process every available listening run independently, optionally in parallel."""
    events_paths = sorted(LISTEN_DIR.glob(EVENTS_GLOB))
    if not events_paths:
        raise FileNotFoundError(
            f'No listening-run events files found under {LISTEN_DIR} '
            f'matching {EVENTS_GLOB!r}'
        )

    pending_paths = [
        events_path for events_path in events_paths
        if not _output_path_for_run(events_path).is_file()
    ]
    skipped = len(events_paths) - len(pending_paths)
    print(
        f'{skipped} of {len(events_paths)} runs already processed; '
        f'{len(pending_paths)} to process.'
    )

    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_run_safely, events_path): events_path.name.removesuffix('_events.tsv')
            for events_path in pending_paths
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Computing binary-confidence activation',
            unit='run',
        ):
            run_prefix = futures[future]
            try:
                error = future.result()
            except BrokenProcessPool:
                # The same exception is set on every pending future; a traceback adds nothing.
                error = (
                    'A worker process died abruptly, most likely killed for running out of memory. '
                    'Re-run with fewer --workers.'
                )
            except Exception:
                error = traceback.format_exc()
            if error is not None:
                failures[run_prefix] = error
                tqdm.write(f'FAILED {run_prefix}:\n{error}')

    if failures:
        print(f'{len(failures)} run(s) failed; re-run the script to retry them: {sorted(failures)}')
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Parallel runs. Each worker needs several GB of RAM (raw data, Maxwell filtering, epochs), '
             'so keep workers * per-worker RAM well below free memory.',
    )
    args = parser.parse_args()
    run_binary_whisper_base_delta_activation(workers=args.workers)


if __name__ == '__main__':
    main()
