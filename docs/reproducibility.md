# Reproducibility notes

## Canonical workflow

The original pipeline used YOLO11 pose models for dorsal, head, and profile images, plus a YOLO ruler detector. The final pose models used `yolo11m-pose.pt`; the ruler detector used `yolo11n.pt`. Pose-model tuning ran for 200 GPU trials, and final pose training ran for 150 epochs with Ultralytics 8.3.202.

Background removal is required at both training and inference time. `scripts/background_remove.py` applies `briaai/RMBG-1.4`, retaining foreground pixels where the mask exceeds 128 and compositing all other pixels over RGB magenta `(255, 0, 255)`.

## Training data and weights

The [Zenodo archive](https://doi.org/10.5281/zenodo.21651735) covers training annotations, split manifests, supporting metadata, and trained weights. The Data Descriptor documents availability of the production measurements.

Code and model weights are distributed under AGPL-3.0. Training annotations and split manifests are distributed under CC0 1.0; the mixed licence scope is stated in the Zenodo package README and per-file licence notices.

## Known sources of variation

- Ultralytics training can vary across GPU hardware and dependency versions.
- RMBG-1.4 and Tesseract outputs can vary across software versions.
- The historical scale-reading workflow used Tesseract and external language models. To calculate traits from new predictions, provide dimensions and scale-bar readings for the corresponding images.
- Source images may change at their upstream host. The split manifest records the original source filenames.

## Release gate

Before publishing, confirm that the repository contains no private collaborator information, raw exports, weights, proprietary inputs, model/API identifiers that should remain confidential, absolute local or cluster paths, or credentials. Run the offline test suite from a clean clone and replace every `pending` placeholder in public metadata.
