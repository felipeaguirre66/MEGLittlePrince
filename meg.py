from datetime import datetime
from time import perf_counter

import mne
import numpy as np
import pandas as pd

from config import LISTEN_DIR


class Meg:
    def __init__(self, run01_prefix: str, filter: bool = True, maxwell: bool = False):
        subject = run01_prefix.split('_')[0]
        meg_dir = LISTEN_DIR / subject / 'ses-01' / 'meg'

        self.raw = mne.io.read_raw_fif(
            meg_dir / f"{run01_prefix}_meg.fif", preload=maxwell, allow_maxshield=True, verbose=False
        )
        self.sfreq = self.raw.info['sfreq']

        self.noisy_channels: list[str] = []
        self.flat_channels: list[str] = []
        self.raw_before_maxwell: mne.io.BaseRaw | None = None
        if maxwell:
            cross_talk = meg_dir / f"{subject}_ses-01_acq-crosstalk_meg.fif"
            calibration = meg_dir / f"{subject}_ses-01_acq-calibration_meg.dat"

            started_at = datetime.now()
            started = perf_counter()
            print(f"Bad channel detection started at {started_at:%Y-%m-%d %H:%M:%S}.")
            self.noisy_channels, self.flat_channels = mne.preprocessing.find_bad_channels_maxwell(
                self.raw, cross_talk=cross_talk, calibration=calibration, verbose=False
            )
            self.raw.info['bads'] = self.noisy_channels + self.flat_channels
            elapsed = perf_counter() - started
            ended_at = datetime.now()
            print(
                f"Bad channel detection ended at {ended_at:%Y-%m-%d %H:%M:%S}; "
                f"elapsed {elapsed:.2f} seconds. "
                f"Noisy: {self.noisy_channels}; flat: {self.flat_channels}."
            )

            started_at = datetime.now()
            started = perf_counter()
            print(f"Maxwell filtering started at {started_at:%Y-%m-%d %H:%M:%S}.")
            self.raw_before_maxwell = self.raw
            self.raw = mne.preprocessing.maxwell_filter(
                self.raw, cross_talk=cross_talk, calibration=calibration,
                coord_frame='head', verbose=False
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

    def get_gradio(self, start: int | None = None, stop: int | None = None) -> np.ndarray:
        """Gradiometer signal (204, n_samples) in T/m; start/stop are sample indices on raw.times."""
        if self._filtered_grads is not None:
            return self._filtered_grads[:, start:stop].copy()
        return self.raw.get_data(picks=self.grad_picks, start=start, stop=stop)

    def get_gradio_by_word(
        self,
        word_idx: int,
        before_start_ms: float = -500,
        after_start_ms: float = 800,
        remove_outlier_channels_by_z_score: bool = True,
    ) -> np.ndarray:
        """Gradiometer signal from `before_start_ms` to `after_start_ms` around the onset of events row `word_idx`.

        If `remove_outlier_channels_by_z_score`, drops channels whose window mean is >= 4 std
        away from the mean of the channel means.
        """
        onset_sample = round(self.events.loc[word_idx, 'onset'] * self.sfreq)
        start = onset_sample + round(before_start_ms * self.sfreq / 1000)
        stop = onset_sample + round(after_start_ms * self.sfreq / 1000)
        if start < 0 or stop > self.raw.n_times:
            raise ValueError(f"Window [{start}, {stop}) is outside the raw ({self.raw.n_times} samples)")
        data = self.get_gradio(start, stop)
        if remove_outlier_channels_by_z_score:
            channel_means = data.mean(axis=1)
            z = (channel_means - channel_means.mean()) / channel_means.std()
            data = data[np.abs(z) < 4]
        return data
