# Ants Pose Detection

This repository provides the code for estimating anatomical keypoints and measuring ant traits from dorsal, head, and profile images.

The workflow uses `briaai/RMBG-1.4` for background removal, three pose models for keypoint detection, and a ruler detector for locating scale bars. The measurement script combines keypoint predictions with image dimensions and scale-bar readings to calculate traits in millimetres.

## What is in this release

- Portable scripts for background removal, pose/ruler inference, and measurement construction.
- The canonical keypoint schemas and training parameters used for the four final models.
- Documentation of the public data/weight deposit and the pipeline's reproducibility limits.

## Data and weights

The [Zenodo archive](https://doi.org/10.5281/zenodo.21651735) is for the training annotations, dataset split manifests, model configurations, final training arguments, and four trained checkpoints. The accompanying Data Descriptor documents the production trait dataset.

## Quick start

```bash
conda env create -f environment.yml
conda activate ant-pose
python -m unittest discover -s tests
```

### Run pose inference

Extract the Zenodo archive to `data/model-archive/` so that it contains `weights/dorsal_pose.pt`. Place dorsal images in `data/images/dorsal/`. The background-removal step composites each image over the magenta background used during training.

```bash
python scripts/background_remove.py --input-dir data/images/dorsal --output-dir work/dorsal_bgrm
python scripts/run_inference.py pose \
  --view dorsal \
  --weights data/model-archive/weights/dorsal_pose.pt \
  --input-dir work/dorsal_bgrm \
  --output-dir work/dorsal_labels
```

### Convert predictions to measurements

For your own images, use the prediction files in `work/dorsal_labels/labels/` and supply two CSVs. The dimensions CSV gives `image_name`, `width`, and `height` in pixels. The calibration CSV gives `image_name`, the scale bar's physical length (`final_value`, `final_unit`), and its length in pixels (`pixel_width`). Use matching image filenames in both tables and the corresponding filename stems for the YOLO labels.

Try the conversion with a single synthetic prediction and scale bar.

```bash
mkdir -p work/example/labels
printf '%s\n' '0 0.5 0.5 1 1 0.1 0.5 2 0.3 0.5 2 0.7 0.5 2 0.9 0.5 2' > work/example/labels/sample.txt
printf '%s\n' 'image_name,width,height' 'sample.jpg,100,50' > work/example/dimensions.csv
printf '%s\n' 'image_name,final_value,final_unit,pixel_width' 'sample.jpg,1,mm,20' > work/example/calibration.csv
python scripts/build_measurements.py \
  --view dorsal \
  --labels-dir work/example/labels \
  --dimensions-csv work/example/dimensions.csv \
  --calibration-csv work/example/calibration.csv \
  --output-csv work/example/measurements.csv
```

The example produces a pronotum width of 2 mm and a body length of 4 mm.

## Licences

Code and trained weights are released under the [GNU Affero General Public License v3.0](LICENSE), consistent with the Ultralytics YOLO licensing terms. The training annotations and split manifests in the Zenodo record are dedicated to the public domain under CC0 1.0. AntWeb images are not redistributed.

## Citation

See [CITATION.cff](CITATION.cff) for the software citation and the associated archive DOI.
