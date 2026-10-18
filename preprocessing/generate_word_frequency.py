"""Generate French Zipf features for the nine shared word-metadata files.

Run from the repository root with ``python -m preprocessing.generate_word_frequency``.
"""

from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

from config import LISTEN_DIR, RESULTS_DIR

MEDIAN_ZIPF_FREQUENCY = 6.2

RUNS = range(1, 10)
WORD_FREQUENCY_DIR = RESULTS_DIR / 'word_frequency'


def generate_word_frequency_files() -> list[Path]:
    """Write each run's word metadata with French Zipf frequencies added."""
    sourcedata_dir = LISTEN_DIR / 'sourcedata'
    input_paths = [
        sourcedata_dir / f'task-listen_run-{run:02d}_extra_info.tsv'
        for run in RUNS
    ]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            'Required word metadata files not found: '
            + ', '.join(str(path) for path in missing_paths)
        )

    WORD_FREQUENCY_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for input_path in input_paths:
        filename = input_path.name
        words = pd.read_csv(input_path, sep=',', dtype={'word': str})
        if 'word' not in words.columns:
            raise ValueError(f"Word metadata file has no 'word' column: {input_path}")
        if words['word'].isna().any():
            raise ValueError(f"Word metadata file contains missing words: {input_path}")

        words['zipf_frequency'] = words['word'].map(
            lambda word: zipf_frequency(word, 'fr')
        )
        output_path = WORD_FREQUENCY_DIR / filename
        words.to_csv(output_path, sep=',', index=False)
        output_paths.append(output_path)
        print(f'Wrote {output_path}')

    return output_paths


if __name__ == '__main__':
    generate_word_frequency_files()
