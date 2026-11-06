"""Generate synthetic restaurant files locally; retained filename supports existing commands."""
import argparse
from pathlib import Path
import runpy


def download(destination):
    generator = Path(__file__).resolve().parents[4] / 'synthetic-data/engine_hospitality/generate.py'
    return runpy.run_path(str(generator))['generate']('restaurant', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    download(args.destination)
