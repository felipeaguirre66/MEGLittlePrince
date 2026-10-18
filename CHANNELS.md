# Bad channel detection: files and how-to

Goal: reproduce the paper's step "Automated bad channel detection with the Maxwell filtering pipeline (noisy channels via crosstalk + calibration files; flat channels via lack of variance)".

Dataset root: `LISTEN_DIR` / `READ_DIR` in [config.py](config.py) (BIDS, OpenNeuro ds007523 / ds007524).

## Files that can be used

Per subject, in `sub-XX/ses-01/meg/` (one copy per session, shared by all runs):

| File | Role |
|---|---|
| `sub-XX_ses-01_acq-crosstalk_meg.fif` | Maxwell cross-talk (SSS) file -> `cross_talk=` |
| `sub-XX_ses-01_acq-calibration_meg.dat` | Elekta fine-calibration file (per-sensor position/orientation/gain; text, one line per sensor) -> `calibration=` |
| `sub-XX_ses-01_task-<task>_run-NN_meg.fif` | Raw data to analyse (MaxShield data, needs `allow_maxshield=True`) |
| `sub-XX_ses-01_coordsystem.json` | Head coil / fiducial coordinates (not needed for bad-channel detection) |

Both session-level files exist for sub-01 and are already referenced in the notebooks `[2026-09-30] run first dive.ipynb` and `[2026-09-20] first dive.ipynb`.

## Files that do NOT help

- `*_channels.tsv`: every MEG channel has `status = good` (181069 "good" rows across all subjects/runs; the rest are header lines). No bad-channel info is provided.
- `*_meg.json`: only counts and acquisition metadata (306 MEG, 2 EOG, 1 ECG, 18 MISC, 19 TRIG; 1000 Hz; cHPI on).
- No `derivatives/`, no `*bads*`, no MaxFilter output exists in the dataset. The paper's bad-channel list is not published, so it must be recomputed.
- Raw `info['bads']` in the fif headers is empty (sub-01 run-01).

## How to do it (MNE-Python)

`mne.preprocessing.find_bad_channels_maxwell` implements exactly the described method. It returns three values when `return_scores=True`.

```python
import mne
from config import LISTEN_DIR

meg_dir = LISTEN_DIR / 'sub-01' / 'ses-01' / 'meg'
raw = mne.io.read_raw_fif(meg_dir / 'sub-01_ses-01_task-listen_run-01_meg.fif',
                          allow_maxshield=True, preload=True)

noisy, flat, scores = mne.preprocessing.find_bad_channels_maxwell(
    raw,
    cross_talk=meg_dir / 'sub-01_ses-01_acq-crosstalk_meg.fif',
    calibration=meg_dir / 'sub-01_ses-01_acq-calibration_meg.dat',
    return_scores=True,
)
raw.info['bads'] = noisy + flat
```

Optionally apply the full Maxwell filter afterwards (interpolates the bads):

```python
raw_sss = mne.preprocessing.maxwell_filter(
    raw, cross_talk=..., calibration=..., coord_frame='head')
```

Notes:
- Run it per run (bads can differ), or take the union across the 9 runs of a session to keep a consistent channel set.
- The detection needs the data loaded; with `preload=False` it reads lazily and is slow (full 623 s run, 5 s windows).
- Planar gradiometers + magnetometers are all used by SSS, even if you later analyse only gradiometers (see `Meg.grad_picks` in [meg.py](meg.py)).

## Result of a test on sub-01 run-01

Run with the crosstalk + calibration files above:

- Static noisy channels: `MEG0513`, `MEG1423`, `MEG2342`
- Static flat channels (flat in every 5 s interval): `MEG0213`, `MEG0311`, `MEG0813`, `MEG1243`, `MEG1443`, `MEG1541`, `MEG1811`, `MEG2121`, `MEG2122`, `MEG2123`

The 10 flat channels are unusually many and constant over the whole run. Check them before trusting the result (e.g. plot their variance versus neighbours); the default `h_freq`/flat thresholds (`min_count`, `flat` in `find_bad_channels_maxwell`) can be tuned. Note that the raw recording is MaxShield (internal active shielding) data, which can alter amplitudes of some channels.

## Cached detection

Run `python preprocessing/bad_channels_detection.py [--workers N]` to detect bad channels for each run with a progress bar. BLAS threads are pinned to 1 (≈31 s per run versus ≈400 s with default threading, same result). Each worker peaks at ~5 GB of RAM, so the default is 1 worker; use `--workers 2` only with ≥12 GB free. Failed runs are reported at the end and retried on the next invocation. Valid results are stored as `bad_channels.json` under `PREPROCESSING_DIR/sub-XX/<run-prefix>/meg/`; runs with valid caches are skipped, so interrupted batch runs can be resumed.

The cache records noisy and flat channels separately, along with the run and input-file metadata and MNE version. A missing, malformed, or stale cache is recomputed. `Meg(..., detect_bad_channels=True)` performs detection and saves the result without Maxwell filtering; `Meg(..., maxwell=True)` loads the cache when valid (otherwise warns and detects), marks the channels bad, and then applies Maxwell filtering.
