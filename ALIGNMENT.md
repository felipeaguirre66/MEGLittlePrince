# MEG / events alignment report (Pallier 2025, listening dataset `ds007523`)

Dataset root: `LISTEN_DIR` (see `config.py`). 58 subjects, 1 session, 9 runs each (one run per audiobook chapter group).

## TL;DR

- `events.tsv` and `_meg.fif` **are aligned**, but the alignment is not visible by comparing durations.
- `events.tsv` onsets are on the **absolute FIF clock**, i.e. they *include* `raw.first_samp`. `raw.times` starts at 0, so
  `onset` and `raw.times` are **not** on the same clock. Offset between them = `raw.first_samp / sfreq` (7 s to 352 s in the surveyed runs).
- The duration check `(last onset + duration - first onset) != raw duration` is expected to fail, even with perfect alignment:
  the raw file contains silence/rest before the first word and after the last word (span of events was 0.13-0.90 min shorter than the raw in every surveyed run).
- Correct use: `sample = round(onset * sfreq)` is a sample index of the **full** raw (`raw.first_samp` included). In MNE, use `mne.Annotations(onset=events.onset - raw.first_samp/sfreq, ...)` or build `events` array with `onset*sfreq` directly (MNE `events[:,0]` is also absolute, includes `first_samp`).
- Do **not** do `raw.time_as_index(onset)` or `raw.crop(tmin=onset)`; both treat `onset` as relative to the start of the file and will be off by `first_samp` (and may fall outside the file).

## Evidence

Surveyed: subjects 1, 2, 3, 10, 20, 30, 40, 50, 58 (all 9 runs each). Script logic: for each run, load FIF header (`preload=False`), `events.tsv`, `sourcedata/task-listen_run-XX_extra_info.tsv`, `infos.csv`, and detect triggers from `STI101`.

### 1. Clock of `events.tsv`

`sub-01`, run 01:

| Quantity | Value |
|---|---|
| `raw.info['sfreq']` | 1000 Hz |
| `raw.first_samp` | 85000 (85.0 s) |
| `raw.times[-1]` | 622.999 s (10.38 min) |
| absolute range of the raw | 85.0 s - 708.0 s |
| first event onset | 90.175 s |
| last event end | 697.499 s |
| events span | 10.12 min |

Events lie inside the *absolute* range [85, 708] s, not inside `raw.times` [0, 623] s. This holds for every surveyed run
(`last event end - (first_samp + duration) < 0` always; max -2.99 s). For several subjects (e.g. sub-10 run 6: first word at 382 s, raw duration 783 s)
the onsets are far above what a relative clock would allow for early words, and some words end after `raw.times[-1]` if read as relative.

### 2. Cross-check against the trigger channel

`STI101` contains one pulse (value 128) per word-related event (2826 pulses for 1614 words in sub-01 run 01; `STI001` marks start/end of the audio, value 5, at samples 86949 and 700533).
Using the absolute clock, every event onset has a trigger within 18 ms (sub-01 run 01: median 16 ms, max 18 ms, 100 % within 20 ms).
Subjects 1, 10, 20, 30, 50, 58: median distance 8-17 ms, which is the known constant audio/trigger latency, consistent with a good alignment.

Caveats:
- Subject 2: median is 8 ms, but max distance is 0.17-0.41 s, so some events do not match a nearby trigger (my naive nearest-trigger check; needs a proper matching).
- Subjects 3 and 40: trigger codes differ (`STI101` has 129 for sub-03; 2049/2176 for sub-40), so the value-128 filter returned nothing. Event codes are not uniform across subjects and should be inspected per subject.

### 3. `sourcedata/task-listen_run-XX_extra_info.tsv` (word-level metadata)

One file per run (9 files), **audio-relative** onsets (first word at 3.05 s in run 01), extra columns `word, n_closing, is_last_word, pos, content_word, sequence_id`.
It has more rows than `events.tsv` (run 01: 1632 vs 1614; every run has +18). Matching `events.stimulus` to `extra_info.word` (case-insensitive sequence match) assigns **all** events to a word of `extra_info`.
Linear fit `events.onset = a * extra_info.onset + b` (sub-01 run 01): `a = 0.9999815`, `b = 87.130`, max residual 5 ms.

### 4. `sourcedata/stimuli/offset_stretch/infos.csv` (the alignment parameters)

Columns: `subject, run, offset, stretch` (487 rows, subjects 23, 57, 58 are missing; 55 subjects).

- `stretch` is the clock-drift factor between audio time and MEG time. It matches the fitted slope: mean absolute difference 2.3e-5, max 4.8e-4 over the 54 runs checked
  (e.g. sub-02 run 01: stretch 1.0017 vs fitted 1.0017; sub-01 run 01: 0.9999934 vs 0.9999815).
- `offset` is in **milliseconds** and is the latency of the first word from the start of the raw (`raw.times`), not from the absolute clock.
  Sub-01 run 01: offset 5153.95 ms; first trigger 90157 - first_samp 85000 = 5157 ms.
  Correlation between `offset` and `(fitted intercept - first_samp)` is 0.9999.
- Empirical formula that reproduces `events.onset` (54 runs checked, residuals in seconds):

  ```
  events.onset ~= raw.first_samp/sfreq + offset/1000 + stretch * (extra_info.onset - extra_info.onset[first word])
  ```

  Median residual -0.06 s on average (std 0.10 s; worst 0.35 s, which is run 9 in all of subjects 2, 10, 20, 30, 50 with a nearly constant -0.34 s, so run 9 seems to have a slightly different reference word).
  Without subtracting the first word onset (3.05 s) the residual is a constant -3.1 s, confirming that `offset` refers to the first word.

So `events.tsv` was obtained by: audio word onsets (`extra_info`) -> stretched and shifted with `infos.csv` -> expressed on the FIF absolute clock. The paper's description (event-channel onsets matched to metadata word onsets with a tolerance) corresponds to the same `offset/stretch` fit (the 5 ms residual).

## Other files that could be relevant

| File | Content | Use |
|---|---|---|
| `sourcedata/stimuli/offset_stretch/infos.csv` | per subject/run `offset` (ms) and `stretch` | audio -> MEG time conversion |
| `sourcedata/task-listen_run-XX_extra_info.tsv` | word-level audio-relative onsets + POS, content-word flag, sequence id | word features for epochs |
| `sourcedata/stimuli/audio/chX-Y.wav` | 9 audio segments | reference time axis |
| `sourcedata/stimuli/phonemes/chX-Y.TextGrid` | forced-alignment of phonemes | phoneme/word timing (audio clock) |
| `sourcedata/stimuli/txt_laser/run{1-7}.txt`, `txt_raw`, `syntax*` | text versions / syntax | text-level features only (no timing) |
| `sub-XX/ses-01/meg/*_meg.fif`, channel `STI101`, `STI001` | triggers (128 = word, STI001 = audio start/end) | independent check of events |
| `sub-XX/ses-01/sub-XX_ses-01_scans.tsv` | acquisition time per run | not needed for sample-level alignment |
| `task-listen_events.json` | only describes `trial_type` and `stimulus` | none |

## Recommendations

1. Treat `events.tsv` onsets as absolute (`first_samp`-inclusive); convert with `onset - raw.first_samp/sfreq` before using relative APIs.
2. Join `events.tsv` with `extra_info.tsv` on word order (sequence matching, as extra_info has 18 extra rows per run) to bring in POS/frequency features.
3. If sub-sample timing is needed, refine with `STI101` triggers (constant ~16 ms trigger-to-event shift for most subjects), after checking each subject's trigger codes (subjects 3 and 40 differ; subject 2 has outliers).
4. Items not verified: the full 58x9 survey (only 9 subjects done), the reason for run-9 constant residual, and the meaning of the 2826 vs 1614 pulse count.
