#!/usr/bin/env python3
"""Train a final pose or ruler model using the recorded release parameters."""

import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    pose = subparsers.add_parser("pose")
    pose.add_argument("--view", choices=("dorsal", "head", "profile"), required=True)
    ruler = subparsers.add_parser("ruler")
    for command in (pose, ruler):
        command.add_argument("--data-yaml", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--weights", default=None, help="Base Ultralytics checkpoint; defaults to recorded model")
        command.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    parameter_file = Path(__file__).parents[1] / "configs" / "training_parameters.yaml"
    parameters = yaml.safe_load(parameter_file.read_text(encoding="utf-8"))
    if args.task == "pose":
        train_args = dict(parameters["common_pose"])
        train_args.update(parameters["views"][args.view])
        default_weights = train_args.pop("model")
        name = f"train_{args.view}_view"
    else:
        train_args = dict(parameters["ruler"])
        default_weights = train_args.pop("model")
        train_args.pop("task", None)
        name = "train_ruler_detect"

    from ultralytics import YOLO

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights or default_weights)
    model.train(
        data=args.data_yaml,
        project=args.output_dir.parent,
        name=args.output_dir.name or name,
        exist_ok=True,
        device=args.device,
        **train_args,
    )


if __name__ == "__main__":
    main()
