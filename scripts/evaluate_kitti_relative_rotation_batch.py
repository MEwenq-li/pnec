import argparse
import csv
from math import acos
from pathlib import Path

import numpy as np


SEQUENCES = [f"{i:02d}" for i in range(11)]


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_angle(R: np.ndarray) -> float:
    value = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return acos(value)


def load_est_poses(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(row[4], row[5], row[6], row[7])
        T[:3, 3] = row[1:4]
        poses.append(T)
    return poses


def load_gt_poses(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, :4] = row.reshape(3, 4)
        poses.append(T)
    return poses


def relative_sequence(poses: list[np.ndarray]) -> list[np.ndarray]:
    return [np.linalg.inv(poses[i]) @ poses[i + 1] for i in range(len(poses) - 1)]


def compose_window(rel_poses: list[np.ndarray], start: int, distance: int) -> np.ndarray:
    T = np.eye(4)
    for idx in range(start, start + distance):
        T = T @ rel_poses[idx]
    return T


def rmse_rotation_windows(gt_rel: list[np.ndarray], est_rel: list[np.ndarray], distance: int) -> float:
    n = min(len(gt_rel), len(est_rel))
    gt_rel = gt_rel[:n]
    est_rel = est_rel[:n]
    errors = []
    for start in range(n - distance + 1):
        gt_win = compose_window(gt_rel, start, distance)
        est_win = compose_window(est_rel, start, distance)
        errors.append(rot_angle(gt_win[:3, :3].T @ est_win[:3, :3]))
    return float(np.sqrt(np.mean(np.square(errors))))


def mean_translation_direction_error(gt_rel: list[np.ndarray], est_rel: list[np.ndarray]) -> float:
    n = min(len(gt_rel), len(est_rel))
    errors = []
    for i in range(n):
        gt_t = gt_rel[i][:3, 3]
        est_t = est_rel[i][:3, 3]
        gt_n = np.linalg.norm(gt_t)
        est_n = np.linalg.norm(est_t)
        if gt_n == 0 or est_n == 0:
            continue
        cos_pos = np.clip(np.dot(gt_t, est_t) / (gt_n * est_n), -1.0, 1.0)
        cos_neg = np.clip(np.dot(-gt_t, est_t) / (gt_n * est_n), -1.0, 1.0)
        errors.append(min(acos(cos_pos), acos(cos_neg)))
    return float(np.mean(errors))


def evaluate_sequence(est_path: Path, gt_path: Path) -> dict[str, float]:
    est_abs = load_est_poses(est_path)
    gt_abs = load_gt_poses(gt_path)
    n = min(len(est_abs), len(gt_abs))
    est_abs = est_abs[:n]
    gt_abs = gt_abs[:n]

    est_rel = relative_sequence(est_abs)
    gt_rel = relative_sequence(gt_abs)
    rel_n = min(len(est_rel), len(gt_rel))
    est_rel = est_rel[:rel_n]
    gt_rel = gt_rel[:rel_n]

    rpe1 = rmse_rotation_windows(gt_rel, est_rel, 1)

    rpen_terms = []
    for distance in range(1, rel_n):
        rpen_terms.append(rmse_rotation_windows(gt_rel, est_rel, distance))
    rpen = float(np.mean(rpen_terms))

    r_t = mean_translation_direction_error(gt_rel, est_rel)

    return {
        "frames_abs": n,
        "frames_rel": rel_n,
        "RPE1_rel_deg": rpe1 * 180.0 / np.pi,
        "RPEn_rel_deg": rpen * 180.0 / np.pi,
        "r_t_rel_deg": r_t * 180.0 / np.pi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KITTI runs directly on relative-pose sequences.")
    parser.add_argument("--results-root", required=True, help="Root directory containing XX_nec result folders.")
    parser.add_argument("--gt-root", required=True, help="KITTI sequences root directory.")
    parser.add_argument("--suffix", default="_nec", help="Result folder suffix, default _nec.")
    parser.add_argument("--output-csv", required=True, help="CSV path for summary table.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq in SEQUENCES:
        est_path = results_root / f"{seq}{args.suffix}" / "rot_avg" / "poses.txt"
        gt_path = gt_root / seq / "poses.txt"
        if not est_path.is_file() or not gt_path.is_file():
            print(f"Skipping {seq}: missing files")
            continue

        metrics = evaluate_sequence(est_path, gt_path)
        rows.append({"sequence": seq, **metrics})
        print(
            f"{seq}: RPE1_rel={metrics['RPE1_rel_deg']:.6f} deg, "
            f"RPEn_rel={metrics['RPEn_rel_deg']:.6f} deg, "
            f"r_t_rel={metrics['r_t_rel_deg']:.6f} deg"
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "frames_abs", "frames_rel", "RPE1_rel_deg", "RPEn_rel_deg", "r_t_rel_deg"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary_csv={output_csv}")


if __name__ == "__main__":
    main()
