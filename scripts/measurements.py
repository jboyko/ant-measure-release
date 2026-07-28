"""Shared parsing and trait-calculation functions for the public release."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

KEYPOINTS = {
    "dorsal": ["head", "pw_left", "pw_right", "tail"],
    "head": ["back", "front", "left", "mandible_root", "mandible_tip", "right"],
    "profile": ["Gaster", "Head", "Mesoma", "Node", "Sting", "Weber_extra"],
}
TRAITS = {
    "dorsal": {"pronotum_width": ("pw_right", "pw_left"), "body_length": ("head", "tail")},
    "head": {
        "head_width": ("left", "right"),
        "head_length": ("back", "front"),
        "mandible_length": ("mandible_root", "mandible_tip"),
    },
    "profile": {"weber_length": ("Weber_extra", "Node"), "head_length_profile": ("Head", "Mesoma")},
}


def image_stem(name: str) -> str:
    return Path(str(name)).stem


def normalize_unit(value, unit):
    """Return a scale value in millimetres, or NaN for an unknown unit."""
    if pd.isna(value) or pd.isna(unit):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    text = str(unit).strip()
    if re.fullmatch(r"[uµμ]m", text, flags=re.IGNORECASE):
        return value / 1000.0
    if re.search(r"m{2}|mi[mn]", text, flags=re.IGNORECASE):
        return value
    if re.fullmatch(r"cm", text, flags=re.IGNORECASE):
        return value * 10.0
    return value / 1000.0 if value > 99 else np.nan


def read_yolo_keypoints(labels_dir: Path, dimensions: pd.DataFrame, view: str) -> pd.DataFrame:
    dimension_names = dimensions.copy()
    dimension_names["_stem"] = dimension_names["image_name"].map(image_stem)
    by_stem = dimension_names.set_index("_stem")
    expected = len(KEYPOINTS[view])
    records = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        line = label_path.read_text(encoding="utf-8").splitlines()
        if not line:
            continue
        try:
            raw = np.asarray([float(value) for value in line[0].split()[5:]], dtype=float).reshape(-1, 3)
        except ValueError:
            continue
        if len(raw) < expected or label_path.stem not in by_stem.index:
            continue
        dim = by_stem.loc[label_path.stem]
        records.append(
            {
                "image_name": dim["image_name"],
                "width": float(dim["width"]),
                "height": float(dim["height"]),
                "keypoints": raw[:expected, :2],
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    calibration = calibration.copy()
    if "image_name" not in calibration.columns and "image" in calibration.columns:
        calibration.rename(columns={"image": "image_name"}, inplace=True)
    required = {"image_name", "final_value", "final_unit", "pixel_width"}
    missing = required.difference(calibration.columns)
    if missing:
        raise ValueError(f"Calibration table is missing columns: {', '.join(sorted(missing))}")
    calibration["_stem"] = calibration["image_name"].map(image_stem)
    calibration["final_value_mm"] = [
        normalize_unit(value, unit) for value, unit in zip(calibration["final_value"], calibration["final_unit"])
    ]
    calibration["pixel_width"] = pd.to_numeric(calibration["pixel_width"], errors="coerce")
    return calibration[["_stem", "final_value_mm", "pixel_width"]].drop_duplicates("_stem", keep="last")


def build_measurements(labels_dir: Path, dimensions: pd.DataFrame, calibration: pd.DataFrame, view: str) -> pd.DataFrame:
    if view not in KEYPOINTS:
        raise ValueError(f"Unknown view: {view}")
    records = read_yolo_keypoints(labels_dir, dimensions, view)
    if records.empty:
        return pd.DataFrame(columns=["image_name"])
    records["_stem"] = records["image_name"].map(image_stem)
    result = records.merge(prepare_calibration(calibration), on="_stem", how="left")
    result["px_per_mm"] = result["pixel_width"] / result["final_value_mm"]
    index = {name: position for position, name in enumerate(KEYPOINTS[view])}
    for trait, (first, second) in TRAITS[view].items():
        def distance(row):
            points = row["keypoints"] * np.asarray([row["width"], row["height"]])
            return float(np.linalg.norm(points[index[first]] - points[index[second]]))
        result[f"{trait}_px"] = result.apply(distance, axis=1)
        result[f"{trait}_mm"] = result[f"{trait}_px"] / result["px_per_mm"]
    columns = ["image_name"]
    columns += [f"{trait}_px" for trait in TRAITS[view]]
    columns += ["final_value_mm", "pixel_width", "px_per_mm"]
    columns += [f"{trait}_mm" for trait in TRAITS[view]]
    return result[columns].sort_values("image_name").reset_index(drop=True)
