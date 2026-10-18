import os

# Must be set before numpy/mne are imported: default multi-threaded BLAS is ~10x slower for this workload.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Allow direct execution as `python preprocessing/bad_channels_detection.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LISTEN_DIR
from meg import Meg, _bad_channel_cache_metadata, _load_bad_channel_cache, bad_channels_cache_path


def detect_run(run_prefix: str) -> str | None:
    """Detect and cache bad channels for one run. Returns an error message, or None on success."""
    warnings.simplefilter('ignore')
    try:
        Meg(run_prefix, filter=False, detect_bad_channels=True)
    except Exception:
        return traceback.format_exc()
    return None


def is_cached(run_path: Path) -> bool:
    run_prefix = run_path.stem[:-4]  # remove "_meg"
    try:
        metadata = _bad_channel_cache_metadata(run_prefix, run_path.parent)
    except OSError:
        return False
    return _load_bad_channel_cache(bad_channels_cache_path(run_prefix), metadata) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Parallel runs. Each worker peaks at ~5 GB of RAM, so keep workers * 5 GB well below free memory.',
    )
    args = parser.parse_args()

    runs = sorted(LISTEN_DIR.glob('sub-*/ses-01/meg/*_run-*_meg.fif'))
    if not runs:
        raise FileNotFoundError(f'No MEG run files found under {LISTEN_DIR}')

    pending = [run_path for run_path in runs if not is_cached(run_path)]
    print(f'{len(runs) - len(pending)} of {len(runs)} runs already cached; {len(pending)} to process.')

    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(detect_run, run_path.stem[:-4]): run_path.stem[:-4] for run_path in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Detecting bad MEG channels', unit='run'):
            run_prefix = futures[future]
            try:
                error = future.result()
            except Exception:  # e.g. worker killed by the OOM killer
                error = traceback.format_exc()
            if error is not None:
                failures[run_prefix] = error
                tqdm.write(f'FAILED {run_prefix}:\n{error}')

    if failures:
        print(f'{len(failures)} run(s) failed; re-run the script to retry them: {sorted(failures)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
