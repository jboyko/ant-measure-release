#!/usr/bin/env python3
"""Apply the required RMBG-1.4/magenta preprocessing to a directory of images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def composite(image: Image.Image, mask) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    background = np.full(rgb.shape, (255, 0, 255), dtype=np.uint8)
    return Image.fromarray(np.where(mask_array[..., None] > 128, rgb, background))


def main():
    args = parse_args()
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    from transformers import pipeline
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in images if args.overwrite or not (args.output_dir / p.name).exists()]
    device = 0 if torch.cuda.is_available() else -1
    remover = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)

    for start in range(0, len(pending), args.batch_size):
        batch_paths = pending[start:start + args.batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        masks = remover(batch_images, return_mask=True)
        if not isinstance(masks, list):
            masks = [masks]
        for path, image, mask in zip(batch_paths, batch_images, masks):
            composite(image, mask).save(args.output_dir / path.name)
        print(f"Processed {min(start + len(batch_paths), len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
