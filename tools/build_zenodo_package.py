#!/usr/bin/env python3
"""Build an audited Zenodo package from the private project archive.

The package intentionally contains only training annotations/splits, public
configuration metadata, and sanitized final model checkpoints. It does not
copy images, raw Labelbox exports, production measurements, notes, or private
working material.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from datetime import date

import torch
import yaml


POSE_DATASETS = {
    "dorsal": ("Dorsal_view", ("train", "val", "test")),
    "head": ("Head_view", ("train", "val", "test")),
    "profile": ("Profile_view", ("train", "val", "test")),
}
RULER_DATASET = ("Ruler_detect", ("train", "val"))
WEIGHTS = {
    "dorsal_pose": "train_dorsal_view",
    "head_pose": "train_head_view",
    "profile_pose": "train_profile_view",
    "ruler_detect": "train_ruler_detect",
}
PRIVATE_MARKERS = (b"/" + b"Users/", b"/" + b"home/", b"/" + b"nfs/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_name(label_path: Path) -> str:
    """YOLO labels and source images share a stem; image bytes are not released."""
    return label_path.with_suffix(".jpg").name


def copy_annotations(project: Path, package: Path) -> list[dict]:
    records = []
    datasets = {**POSE_DATASETS, "ruler": RULER_DATASET}
    for public_name, (dataset_name, splits) in datasets.items():
        for split in splits:
            source = project / "datasets" / dataset_name / "labels" / split
            destination = package / "training-data" / "annotations" / public_name / split
            if not source.is_dir():
                raise FileNotFoundError(f"Missing annotation directory: {source}")
            destination.mkdir(parents=True, exist_ok=True)
            for label in sorted(source.glob("*.txt")):
                copied = destination / label.name
                shutil.copy2(label, copied)
                lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                records.append(
                    {
                        "dataset": public_name,
                        "split": split,
                        "source_image_name": image_name(label),
                        "annotation_path": copied.relative_to(package).as_posix(),
                        "annotation_rows": len(lines),
                        "sha256": sha256(copied),
                    }
                )
    return records


def sanitise_args(args: dict, data_config: str, base_model: str) -> dict:
    clean = dict(args)
    clean["data"] = data_config
    clean["model"] = base_model
    clean.pop("save_dir", None)
    return clean


def sanitise_checkpoint(source: Path, destination: Path, data_config: str, base_model: str) -> None:
    """Remove machine paths from a trusted local Ultralytics checkpoint."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["train_args"] = sanitise_args(checkpoint["train_args"], data_config, base_model)
    for model_key in ("model", "ema"):
        model = checkpoint.get(model_key)
        if model is not None and isinstance(getattr(model, "args", None), dict):
            model.args = sanitise_args(model.args, data_config, base_model)
    torch.save(checkpoint, destination)
    binary = destination.read_bytes()
    leaked = [marker.decode("utf-8", errors="replace") for marker in PRIVATE_MARKERS if marker in binary]
    if leaked:
        destination.unlink()
        raise ValueError(f"Checkpoint still contains private markers: {', '.join(leaked)}")


def copy_weights_and_args(project: Path, package: Path) -> list[dict]:
    weights_dir = package / "weights"
    args_dir = package / "training-arguments"
    weights_dir.mkdir(parents=True, exist_ok=True)
    args_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for public_name, run_name in WEIGHTS.items():
        source_dir = project / "ant_training" / run_name
        source_weight = source_dir / "weights" / "best.pt"
        source_args = source_dir / "args.yaml"
        if not source_weight.is_file() or not source_args.is_file():
            raise FileNotFoundError(f"Missing final artifacts for {run_name}")
        is_ruler = public_name == "ruler_detect"
        config = "configs/ruler_detect.yaml" if is_ruler else f"configs/{public_name.removesuffix('_pose')}_view.yaml"
        base_model = "yolo11n.pt" if is_ruler else "yolo11m-pose.pt"
        args = yaml.safe_load(source_args.read_text(encoding="utf-8"))
        clean_args = sanitise_args(args, config, base_model)
        args_path = args_dir / f"{public_name}_args.yaml"
        args_path.write_text(yaml.safe_dump(clean_args, sort_keys=False), encoding="utf-8")
        weight_path = weights_dir / f"{public_name}.pt"
        sanitise_checkpoint(source_weight, weight_path, config, base_model)
        records.append(
            {
                "model": public_name,
                "checkpoint": weight_path.relative_to(package).as_posix(),
                "checkpoint_sha256": sha256(weight_path),
                "training_arguments": args_path.relative_to(package).as_posix(),
                "training_arguments_sha256": sha256(args_path),
            }
        )
    return records


def copy_release_code(project: Path, package: Path) -> None:
    """Include a source snapshot so code and weights share one archived record."""
    source = project / "public-release"
    destination = package / "code"
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store", "*.pyc"),
    )


def write_documentation(package: Path, annotation_rows: list[dict], weight_rows: list[dict], release_url: str, doi: str) -> None:
    counts = {}
    for row in annotation_rows:
        key = (row["dataset"], row["split"])
        counts[key] = counts.get(key, 0) + 1
    count_lines = "\n".join(f"- `{dataset}/{split}`: {count:,} labels" for (dataset, split), count in sorted(counts.items()))
    (package / "README.md").write_text(
        f"""# Ant Measure Release — training data and model weights

This Zenodo package is archived at DOI [{doi}](https://doi.org/{doi}) and accompanies [{release_url}]({release_url}). It contains the
training annotations and final model weights for the multi-view ant pose-estimation
pipeline. It contains no images, raw Labelbox exports, production measurements,
manual corrections, API requests/responses, or collaborator notes.

## Contents

- `training-data/annotations/`: YOLO-format pose/detection annotations arranged by dataset and split.
- `training-data/split_manifest.csv`: one row per released label file, including its SHA-256 checksum.
- `configs/`: keypoint schema and YOLO dataset configuration.
- `training-arguments/`: canonical final-run arguments with local paths replaced by archive-relative paths.
- `weights/`: final checkpoints with machine-specific checkpoint metadata removed; learned parameters are unchanged.
- `code/`: a snapshot of the public AGPL-3.0 source release.
- `SHA256SUMS`: integrity checks for every included payload file.

## Released annotation counts

{count_lines}

## Licences

- `training-data/**` is CC0 1.0.
- `weights/**`, `training-arguments/**`, and `configs/**` are AGPL-3.0-only.

The mixed licence scope is deliberate: Ultralytics states that trained YOLO models
are AGPL-3.0 by default. See `MODEL_LICENSE_NOTICE.md`.
""",
        encoding="utf-8",
    )
    (package / "DATA_DICTIONARY.md").write_text(
        """# Data dictionary

Each annotation file is plain-text YOLO format. Coordinates are normalized to the
image width and height. Images are not included in this release.

## Pose annotations

Each row is `class x_center y_center width height` followed by `(x, y, visibility)`
for every keypoint. `visibility=2` denotes visible, `1` denotes occluded, and a
missing keypoint is encoded as `0 0 0`.

Keypoint order is defined by the matching file under `configs/`:

- dorsal: head, pw_left, pw_right, tail
- head: back, front, left, mandible_root, mandible_tip, right
- profile: Gaster, Head, Mesoma, Node, Sting, Weber_extra

## Ruler annotations

Each row is `class x_center y_center width height`. Classes are `0=ant` and
`1=ruler`.

## Split manifest

`dataset`, `split`, `source_image_name`, `annotation_path`, `annotation_rows`, and
the SHA-256 checksum of the corresponding annotation file are provided for each
released label file. `source_image_name` identifies the source image but no image
content or image URL is redistributed.
""",
        encoding="utf-8",
    )
    (package / "LICENSE-DATA-CC0.txt").write_text(
        """CC0 1.0 Universal dedication

To the extent possible under law, James D. Boyko has waived all copyright and
related or neighbouring rights to the files under training-data/annotations/ and
training-data/split_manifest.csv. These data are dedicated to the public domain
under CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/
""",
        encoding="utf-8",
    )
    (package / "MODEL_LICENSE_NOTICE.md").write_text(
        """# Model and code licence notice

The checkpoints, configuration files, and training arguments in this package are
released under GNU Affero General Public License version 3 only (AGPL-3.0-only).
They were trained with Ultralytics YOLO. Ultralytics states that trained YOLO models
are covered by AGPL-3.0 by default. See https://www.ultralytics.com/license.
""",
        encoding="utf-8",
    )
    metadata = {
        "title": "Ant Measure Release: training annotations and model weights",
        "upload_type": "dataset",
        "publication_date": str(date.today()),
        "doi": doi,
        "creators": [{"name": "Boyko, James D.", "affiliation": "Department of Ecology and Evolutionary Biology, University of Michigan, Ann Arbor, MI 48109"}],
        "description": "Training annotations, split manifests, metadata, and final model weights for multi-view ant pose estimation. Images and production measurements are not included.",
        "keywords": ["ant morphology", "pose estimation", "computer vision", "morphometrics", "YOLO"],
        "license_note": "Mixed-license record: training-data is CC0-1.0; code/configuration/weights are AGPL-3.0-only."
    }
    (package / "zenodo-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (package / "training-data" / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    with (package / "weights" / "weights_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(weight_rows, handle, indent=2)
        handle.write("\n")


def write_checksums(package: Path) -> None:
    paths = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256(path)}  {path.relative_to(package).as_posix()}" for path in paths]
    (package / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Private ants-pose-detection working archive")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the inspected package")
    parser.add_argument("--release-url", default="https://github.com/jboyko/ant-measure-release")
    parser.add_argument("--doi", required=True, help="Reserved Zenodo DOI for this package version")
    return parser.parse_args()


def main():
    args = parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        package = Path(tmp) / output.name
        package.mkdir()
        annotations = copy_annotations(project, package)
        weights = copy_weights_and_args(project, package)
        copy_release_code(project, package)
        shutil.copytree(project / "public-release" / "configs", package / "configs")
        shutil.copy2(project / "public-release" / "LICENSE", package / "LICENSE-AGPL-3.0.txt")
        write_documentation(package, annotations, weights, args.release_url, args.doi)
        write_checksums(package)
        shutil.move(str(package), str(output))
    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(f"Package: {output}")
    print(f"Archive:  {archive}")


if __name__ == "__main__":
    main()
