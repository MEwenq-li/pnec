from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Pose, pose_row


@dataclass
class KittiStereoCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


def sequence_paths(root: Path, sequence: str) -> tuple[Path, Path, Path, Path, Path]:
    seq_dir = Path(root) / sequence
    return seq_dir / "image_0", seq_dir / "image_1", seq_dir / "times.txt", seq_dir / "calib.txt", seq_dir / "poses.txt"


def list_images(image_dir: Path, ext: str = ".png") -> list[Path]:
    return sorted(Path(image_dir).glob(f"*{ext}"))


def load_timestamps(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _parse_projection(line: str) -> np.ndarray:
    parts = line.replace(":", " ").split()
    return np.asarray([float(v) for v in parts[1:13]], dtype=np.float64).reshape(3, 4)


def load_kitti_stereo_calibration(path: Path) -> KittiStereoCalibration:
    p0 = None
    p1 = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("P0:"):
                p0 = _parse_projection(line)
            elif line.startswith("P1:"):
                p1 = _parse_projection(line)
    if p0 is None or p1 is None:
        raise ValueError(f"Missing P0/P1 in {path}")
    return KittiStereoCalibration(
        fx=float(p0[0, 0]),
        fy=float(p0[1, 1]),
        cx=float(p0[0, 2]),
        cy=float(p0[1, 2]),
        baseline=float(-p1[0, 3] / p1[0, 0]),
    )


def load_gt_poses(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = row.reshape(3, 4)
        poses.append(T)
    return poses


def prepare_output(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    poses_dir = output_dir / "rot_avg"
    poses_dir.mkdir(parents=True, exist_ok=True)
    return poses_dir / "poses.txt"


def write_poses(path: Path, rows: list[tuple[float, Pose]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for timestamp, pose in rows:
            handle.write(pose_row(timestamp, pose))


def write_timing(path: Path, timing_rows: list[dict[str, float]]) -> None:
    header = "ID FrameLoading FeatureCreation NEC-ES IT-ES AVG-IT-ES CERES OPTIMIZATION TOTAL\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for row in timing_rows:
            handle.write(
                f"{int(row['ID'])} {int(row.get('FrameLoading', 0))} "
                f"{int(row.get('FeatureCreation', 0))} {int(row.get('NEC-ES', 0))} "
                f"{int(row.get('IT-ES', 0))} {int(row.get('AVG-IT-ES', 0))} "
                f"{int(row.get('CERES', 0))} {int(row.get('OPTIMIZATION', 0))} "
                f"{int(row.get('TOTAL', 0))}\n"
            )

