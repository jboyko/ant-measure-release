# Ants Pose Detection

This repository is the public code release for the multi-view ant pose-estimation and morphometrics pipeline described in XXX. 

The release supports three pose-estimation views (dorsal, head, and profile), a ruler detector, background removal with `briaai/RMBG-1.4`, and conversion of predicted keypoints and deposited calibration records into millimetre-scale traits.

## What is in this release

- Portable scripts for background removal, pose/ruler inference, and measurement construction.
- The canonical keypoint schemas and training parameters used for the four final models.
- Documentation of the public data/weight deposit and the pipeline's reproducibility limits.

## Data and weights

The versioned Zenodo release is archived at [10.5281/zenodo.21651735](https://doi.org/10.5281/zenodo.21651735).

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

## Licences

Code and trained weights are released under the [GNU Affero General Public License v3.0](LICENSE), consistent with the Ultralytics YOLO licensing terms. The training annotations and split manifests in the Zenodo record are dedicated to the public domain under CC0 1.0. AntWeb images are not redistributed.

## Citation

See [CITATION.cff](CITATION.cff). Replace the placeholder release version and Zenodo DOI when the data deposit and manuscript are final.
