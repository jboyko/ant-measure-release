#!/usr/bin/env python3
"""Build millimetre trait measurements from YOLO labels and calibration records."""

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from measurements import build_measurements


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("dorsal", "head", "profile"), required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--dimensions-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dimensions = pd.read_csv(args.dimensions_csv)
    required = {"image_name", "width", "height"}
    missing = required.difference(dimensions.columns)
    if missing:
        raise SystemExit(f"Dimensions table is missing columns: {', '.join(sorted(missing))}")
    measurements = build_measurements(
        args.labels_dir, dimensions, pd.read_csv(args.calibration_csv), args.view
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(args.output_csv, index=False, float_format="%.4f")
    print(f"Wrote {len(measurements)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
