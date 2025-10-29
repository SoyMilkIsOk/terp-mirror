"""Calibration utilities for mapping detection coordinates to display space."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import yaml


class CalibrationError(RuntimeError):
    """Raised when calibration data cannot be parsed or applied."""


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration flags that control calibration usage at runtime."""

    enabled: bool = False
    file: Optional[Path] = None
    apply_prompt: bool = True
    apply_spinner: bool = True
    apply_result: bool = True


@dataclass(frozen=True)
class CalibrationMapping:
    """Encapsulate an affine or homography transform for screen alignment."""

    matrix: np.ndarray
    kind: str
    frame_size: tuple[int, int]
    screen_size: tuple[int, int]
    source_points: tuple[tuple[float, float], ...]
    target_points: tuple[tuple[float, float], ...]

    def apply(self, point: tuple[float, float]) -> tuple[int, int]:
        """Map ``point`` from frame coordinates to screen coordinates."""

        if len(self.matrix.shape) != 2 or self.matrix.shape[0] != 3 or self.matrix.shape[1] != 3:
            raise CalibrationError("Calibration matrix must be 3x3.")

        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        transformed = self.matrix @ vec
        if abs(transformed[2]) < 1e-9:
            raise CalibrationError("Calibration transform produced invalid homogeneous coordinate.")

        x = transformed[0] / transformed[2]
        y = transformed[1] / transformed[2]
        return int(round(x)), int(round(y))

    def is_compatible(self, frame_size: tuple[int, int], screen_size: tuple[int, int]) -> bool:
        """Return ``True`` when the stored calibration matches the current setup."""

        return self.frame_size == tuple(frame_size) and self.screen_size == tuple(screen_size)

    def to_dict(self) -> dict:
        """Serialize the mapping for storage."""

        return {
            "kind": self.kind,
            "frame_size": [int(self.frame_size[0]), int(self.frame_size[1])],
            "screen_size": [int(self.screen_size[0]), int(self.screen_size[1])],
            "matrix": self.matrix.tolist(),
            "source_points": [list(pt) for pt in self.source_points],
            "target_points": [list(pt) for pt in self.target_points],
            "created": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }


def compute_calibration_mapping(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    frame_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> CalibrationMapping:
    """Return a calibration transform using the provided ``pairs`` of points."""

    if len(pairs) < 2:
        raise CalibrationError("At least two point pairs are required for calibration.")

    src = np.array([pair[0] for pair in pairs], dtype=np.float64)
    dst = np.array([pair[1] for pair in pairs], dtype=np.float64)

    matrix: Optional[np.ndarray] = None
    kind: str

    if len(pairs) >= 4:
        homography, mask = cv2.findHomography(src, dst, method=0)
        if homography is None or mask is None or not mask.any():
            raise CalibrationError("Unable to compute homography from provided points.")
        matrix = homography
        kind = "homography"
    else:
        if len(pairs) == 3:
            affine, inliers = cv2.estimateAffine2D(src, dst)
        else:
            affine, inliers = cv2.estimateAffinePartial2D(src, dst)
        if affine is None or inliers is None or not inliers.any():
            raise CalibrationError("Unable to compute affine transform from provided points.")
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        kind = "affine"

    matrix = matrix.astype(np.float64)

    return CalibrationMapping(
        matrix=matrix,
        kind=kind,
        frame_size=tuple(int(v) for v in frame_size),
        screen_size=tuple(int(v) for v in screen_size),
        source_points=tuple((float(x), float(y)) for x, y in src.tolist()),
        target_points=tuple((float(x), float(y)) for x, y in dst.tolist()),
    )


def save_calibration_mapping(path: Path, mapping: CalibrationMapping) -> None:
    """Write ``mapping`` to ``path`` in YAML format."""

    payload = mapping.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_calibration_mapping(path: Path) -> CalibrationMapping:
    """Load a calibration transform from ``path``."""

    if not path.exists():
        raise CalibrationError(f"Calibration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    try:
        kind = str(data["kind"])
        matrix_data = data["matrix"]
        frame_size = tuple(int(v) for v in data["frame_size"])
        screen_size = tuple(int(v) for v in data["screen_size"])
        source_points = tuple(tuple(float(coord) for coord in pair) for pair in data["source_points"])
        target_points = tuple(tuple(float(coord) for coord in pair) for pair in data["target_points"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise CalibrationError("Calibration file is malformed.") from exc

    matrix = np.array(matrix_data, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise CalibrationError("Calibration matrix must be 3x3.")

    return CalibrationMapping(
        matrix=matrix,
        kind=kind,
        frame_size=frame_size,
        screen_size=screen_size,
        source_points=source_points,
        target_points=target_points,
    )


def map_point_to_screen(
    point: tuple[float, float],
    frame_size: tuple[int, int],
    screen_size: tuple[int, int],
    calibration: Optional[CalibrationMapping],
) -> tuple[int, int]:
    """Map ``point`` to screen coordinates using calibration when available."""

    if calibration is not None and calibration.is_compatible(frame_size, screen_size):
        return calibration.apply(point)

    scale_x = screen_size[0] / frame_size[0]
    scale_y = screen_size[1] / frame_size[1]
    return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))


def iter_point_pairs(
    sources: Iterable[tuple[float, float]],
    targets: Iterable[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Zip ``sources`` and ``targets`` into calibration point pairs."""

    pairs = []
    for src, dst in zip(sources, targets):
        pairs.append((tuple(float(v) for v in src), tuple(float(v) for v in dst)))
    return pairs

