# Ants Pose Detection — reproducibility release

This repository is the public code release for the multi-view ant pose-estimation and morphometrics pipeline described in the accompanying Data Descriptor. It is a clean release of the existing **ants-pose-detection** project, prepared independently of the private working archive.

The release supports three pose-estimation views (dorsal, head, and profile), a ruler detector, background removal with `briaai/RMBG-1.4`, and conversion of predicted keypoints and deposited calibration records into millimetre-scale traits.

## What is in this release

- Portable scripts for background removal, pose/ruler inference, and measurement construction.
- The canonical keypoint schemas and training parameters used for the four final models.
- A small synthetic fixture and an offline smoke test.
- Documentation of the public data/weight deposit and the pipeline's reproducibility limits.

## What is deliberately not in this repository

- AntWeb images, raw Labelbox exports, model weights, production predictions, measurements, debug renders, or API batch files. These belong in the versioned Zenodo deposit, or remain with their original owner.
- Private planning, manuscript drafts, reviewer notes, co-author material, agent instructions, local/cluster paths, credentials, or chat records.

## Data and weights

The paper release will link a versioned Zenodo record here: **DOI pending**.

The initial record will contain the training-data release (training annotations, split manifests, and supporting metadata) and the four trained weights with their canonical training arguments. AntWeb images and the broader AntWeb-derived production measurement records are not part of this initial deposit; their release requires explicit collaborator approval.

## Quick start

```bash
conda env create -f environment.yml
conda activate ant-pose
python -m unittest discover -s tests
```

For an inference run, download the matching model weights from Zenodo and use background-removed images:

```bash
python scripts/background_remove.py --input-dir data/images/dorsal --output-dir work/dorsal_bgrm
python scripts/run_inference.py pose \
  --view dorsal \
  --weights data/weights/dorsal_pose.pt \
  --input-dir work/dorsal_bgrm \
  --output-dir work/dorsal_labels
```

Build measurements from YOLO prediction labels, image dimensions, and a calibration table:

```bash
python scripts/build_measurements.py \
  --view dorsal \
  --labels-dir work/dorsal_labels/labels \
  --dimensions-csv data/image_dimensions.csv \
  --calibration-csv data/dorsal_units_final.csv \
  --output-csv work/dorsal_measurements.csv
```

## Scientific scope and constraints

The release provides two reproducibility levels:

1. **Data-record reproduction:** regenerate trait tables from deposited keypoints/calibration records with `build_measurements.py`.
2. **Model-workflow reproduction:** download the deposited weights, obtain permitted AntWeb images from their public links, apply the required magenta background-removal step, then run inference and measurement construction.

Exact retraining is subject to ordinary GPU/Ultralytics nondeterminism. Re-running the historical OCR/LLM scale-reading workflow can also vary by Tesseract version and model availability; deposited final calibration values are therefore the canonical inputs for recreating the published tables. See [docs/reproducibility.md](docs/reproducibility.md).

To train a pose model from the archived annotations and permitted source images, first place the YOLO dataset under `data/datasets/` (or update the `path` entry in the matching config), then run:

```bash
python scripts/train_model.py pose \
  --view dorsal \
  --data-yaml configs/dorsal_view.yaml \
  --output-dir work/training/dorsal
```

## Licences

Code is released under the [MIT License](LICENSE). The associated factual annotation/measurement data will be dedicated under CC0 in Zenodo. AntWeb image terms remain applicable to the linked images.

## Citation

See [CITATION.cff](CITATION.cff). Replace the placeholder release version and Zenodo DOI when the data deposit and manuscript are final.
