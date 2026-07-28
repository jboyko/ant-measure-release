#!/usr/bin/env python3
"""Run a deposited pose or ruler model without machine-specific paths."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for task in ("pose", "ruler"):
        command = subparsers.add_parser(task)
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--input-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
        command.add_argument("--confidence", type=float, default=0.1 if task == "pose" else 0.25)
        if task == "pose":
            command.add_argument("--view", choices=("dorsal", "head", "profile"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    from ultralytics import YOLO

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = model.predict(
        source=args.input_dir,
        project=args.output_dir.parent,
        name=args.output_dir.name,
        exist_ok=True,
        save=False,
        save_txt=True,
        conf=args.confidence,
        device=args.device,
        stream=True,
    )
    for _ in results:
        pass
    print(f"YOLO labels: {args.output_dir / 'labels'}")


if __name__ == "__main__":
    main()
