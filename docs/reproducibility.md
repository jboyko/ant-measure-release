# Reproducibility notes

## Canonical workflow

The original pipeline used YOLO11 pose models for dorsal, head, and profile images, plus a YOLO ruler detector. The final pose models used `yolo11m-pose.pt`; the ruler detector used `yolo11n.pt`. Pose-model tuning ran for 200 GPU trials, and final pose training ran for 150 epochs with Ultralytics 8.3.202.

Background removal is required at both training and inference time. `scripts/background_remove.py` applies `briaai/RMBG-1.4`, retaining foreground pixels where the mask exceeds 128 and compositing all other pixels over RGB magenta `(255, 0, 255)`.

## Published data record

The initial public deposit will distribute training annotations, split manifests, supporting metadata, and trained weights—not AntWeb images or the broader production measurement records. The repository documentation must be updated with the Zenodo DOI before public release. Any future deposit of AntWeb-derived identifiers, calibration records, or measurements requires explicit collaborator approval.

## Known sources of variation

- Ultralytics training can vary across GPU hardware and dependency versions.
- RMBG-1.4 and Tesseract outputs can vary across software versions.
- The historical scale-reading recovery used external LLM services. It is optional for use of the deposited data, and is not required to recreate the published measurement tables.
- Some source images may be unavailable or changed at their upstream host. Preserve the image manifest and retrieval date in the Zenodo record.

## Release gate

Before publishing, confirm that the repository contains no private collaborator information, raw exports, weights, proprietary inputs, model/API identifiers that should remain confidential, absolute local or cluster paths, or credentials. Run the offline test suite from a clean clone and replace every `pending` placeholder in public metadata.
