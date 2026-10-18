from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import warnings

import mne
import numpy as np
import pandas as pd
import re

from config import LISTEN_DIR, PREPROCESSING_DIR


BAD_CHANNELS_SCHEMA_VERSION = 1


def bad_channels_cache_path(run_prefix: str) -> Path:
    """Return the cache path for a BIDS run prefix."""
    subject = run_prefix.split('_')[0]
    return PREPROCESSING_DIR / subject / run_prefix / 'meg' / 'bad_channels.json'


def _bad_channel_cache_metadata(run_prefix: str, meg_dir: Path) -> dict:
    files = {
        'raw': meg_dir / f'{run_prefix}_meg.fif',
        'cross_talk': meg_dir / f'{run_prefix.split("_")[0]}_ses-01_acq-crosstalk_meg.fif',
        'calibration': meg_dir / f'{run_prefix.split("_")[0]}_ses-01_acq-calibration_meg.dat',
    }
    inputs = {}
    for key, path in files.items():
        stat = path.stat()
        inputs[key] = {
            'name': path.name,
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        }
    return {
        'run_prefix': run_prefix,
        'mne_version': mne.__version__,
        'method': 'find_bad_channels_maxwell',
        'parameters': {},
        'inputs': inputs,
    }


def _load_bad_channel_cache(cache_path: Path, metadata: dict) -> tuple[list[str], list[str]] | None:
    try:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
        if data.get('schema_version') != BAD_CHANNELS_SCHEMA_VERSION or data.get('metadata') != metadata:
            return None
        noisy = data['noisy_channels']
        flat = data['flat_channels']
        if not (
            isinstance(noisy, list) and all(isinstance(channel, str) for channel in noisy)
            and isinstance(flat, list) and all(isinstance(channel, str) for channel in flat)
        ):
            return None
        return noisy, flat
    except (OSError, json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None


def _save_bad_channel_cache(
    cache_path: Path, metadata: dict, noisy_channels: list[str], flat_channels: list[str]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'schema_version': BAD_CHANNELS_SCHEMA_VERSION,
        'metadata': metadata,
        'noisy_channels': noisy_channels,
        'flat_channels': flat_channels,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=cache_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, indent=2)
            temporary_file.write('\n')
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class Meg:
    def __init__(
        self,
        run01_prefix: str,
        filter: bool = True,
        maxwell: bool = True,
        detect_bad_channels: bool = False,
    ):
        subject = run01_prefix.split('_')[0]
        meg_dir = LISTEN_DIR / subject / 'ses-01' / 'meg'

        self.raw = mne.io.read_raw_fif(
            meg_dir / f"{run01_prefix}_meg.fif",
            preload=maxwell or detect_bad_channels,
            allow_maxshield=True,
            verbose=False,
        )
        self.sfreq = self.raw.info['sfreq']

        self.noisy_channels: list[str] = []
        self.flat_channels: list[str] = []
        self.raw_before_maxwell: mne.io.BaseRaw | None = None
        if maxwell or detect_bad_channels:
            cross_talk = meg_dir / f'{subject}_ses-01_acq-crosstalk_meg.fif'
            calibration = meg_dir / f'{subject}_ses-01_acq-calibration_meg.dat'
            cache_path = bad_channels_cache_path(run01_prefix)
            metadata = _bad_channel_cache_metadata(run01_prefix, meg_dir)
            cached_channels = _load_bad_channel_cache(cache_path, metadata)

            started_at = datetime.now()
            started = perf_counter()
            if cached_channels is None:
                reason = 'missing' if not cache_path.exists() else 'invalid or stale'
                warnings.warn(
                    f"Bad-channel cache {reason} at {cache_path}; detecting channels now.",
                    stacklevel=2,
                )
                print(f"Bad channel detection started at {started_at:%Y-%m-%d %H:%M:%S}.")
                self.noisy_channels, self.flat_channels = mne.preprocessing.find_bad_channels_maxwell(
                    self.raw, cross_talk=cross_talk, calibration=calibration, verbose=False
                )
                _save_bad_channel_cache(
                    cache_path, metadata, self.noisy_channels, self.flat_channels
                )
            else:
                self.noisy_channels, self.flat_channels = cached_channels
                print(f"Loaded bad-channel lists from {cache_path}.")
            self.raw.info['bads'] = self.noisy_channels + self.flat_channels
            if cached_channels is None:
                elapsed = perf_counter() - started
                ended_at = datetime.now()
                print(
                    f"Bad channel detection ended at {ended_at:%Y-%m-%d %H:%M:%S}; "
                    f"elapsed {elapsed:.2f} seconds. "
                    f"Noisy: {self.noisy_channels}; flat: {self.flat_channels}."
                )

        if maxwell:
            cross_talk = meg_dir / f'{subject}_ses-01_acq-crosstalk_meg.fif'
            calibration = meg_dir / f'{subject}_ses-01_acq-calibration_meg.dat'
            reference_prefix = re.sub(r'_run-\d+', '_run-05', run01_prefix)
            reference_fif = meg_dir / f'{reference_prefix}_meg.fif'
            started_at = datetime.now()
            started = perf_counter()
            print(f"Maxwell filtering started at {started_at:%Y-%m-%d %H:%M:%S}.")
            self.raw_before_maxwell = self.raw
            self.raw = mne.preprocessing.maxwell_filter(
                self.raw, cross_talk=cross_talk, calibration=calibration,
                coord_frame='head', 
                destination=reference_fif,
                verbose=False
            )
            elapsed = perf_counter() - started
            ended_at = datetime.now()
            print(
                f"Maxwell filtering ended at {ended_at:%Y-%m-%d %H:%M:%S}; "
                f"elapsed {elapsed:.2f} seconds."
            )

        events = pd.read_csv(meg_dir / f"{run01_prefix}_events.tsv", sep='\t')
        # events.tsv onsets are on the acquisition clock (include first_samp); raw.times starts at 0
        events['onset'] = events['onset'] - self.raw.first_samp / self.sfreq
        self.events = events

        self.grad_picks = [
            i for i, (ch, typ) in enumerate(zip(self.raw.ch_names, self.raw.get_channel_types()))
            if ch.startswith("MEG") and typ == 'grad'
        ]
        self._filtered_grads: np.ndarray | None = None
        if filter:
            started_at = datetime.now()
            started = perf_counter()
            print(f"Filtering gradiometer data (0.5–20 Hz) started at {started_at:%Y-%m-%d %H:%M:%S}.")
            grad_data = self.raw.get_data(picks=self.grad_picks)
            self._filtered_grads = mne.filter.filter_data(
                grad_data, self.sfreq, l_freq=0.5, h_freq=20, copy=False, verbose=False
            )
            elapsed = perf_counter() - started
            ended_at = datetime.now()
            print(
                f"Filtering ended at {ended_at:%Y-%m-%d %H:%M:%S}; "
                f"elapsed {elapsed:.2f} seconds."
            )

    def get_grad(self, start: int | None = None, stop: int | None = None) -> np.ndarray:
        """Gradiometer signal (204, n_samples) in T/m; start/stop are sample indices on raw.times."""
        if self._filtered_grads is not None:
            return self._filtered_grads[:, start:stop].copy()
        return self.raw.get_data(picks=self.grad_picks, start=start, stop=stop)

    def get_grad_by_word(
        self,
        word_idx: int,
        before_start_ms: float = -500,
        after_start_ms: float = 800,
    ) -> np.ndarray:
        """Gradiometer signal from `before_start_ms` to `after_start_ms` around the onset of events row `word_idx`.
        """
        onset_sample = round(self.events.loc[word_idx, 'onset'] * self.sfreq)
        start = onset_sample + round(before_start_ms * self.sfreq / 1000)
        stop = onset_sample + round(after_start_ms * self.sfreq / 1000)
        if start < 0 or stop > self.raw.n_times:
            raise ValueError(f"Window [{start}, {stop}) is outside the raw ({self.raw.n_times} samples)")
        data = self.get_grad(start, stop)
        return data
