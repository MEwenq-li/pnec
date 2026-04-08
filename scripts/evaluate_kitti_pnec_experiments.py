import argparse
import csv
from math import acos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEGMENT_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]


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


def est_row_to_pose(row: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(row[4], row[5], row[6], row[7])
    T[:3, 3] = row[1:4]
    return T


def gt_row_to_pose(row: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = row.reshape(3, 4)
    return T


def rot_angle(R: np.ndarray) -> float:
    value = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return acos(value)


def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)
    Xc = X - mean_x
    Yc = Y - mean_y
    cov = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_x = np.mean(np.sum(Xc * Xc, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_x
    t = mean_y - scale * R @ mean_x
    return scale, R, t


def trajectory_distances(poses: list[np.ndarray]) -> np.ndarray:
    distances = [0.0]
    for i in range(1, len(poses)):
        step = np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])
        distances.append(distances[-1] + step)
    return np.asarray(distances)


def last_frame_from_length(distances: np.ndarray, first_frame: int, length: float) -> int:
    idx = np.searchsorted(distances, distances[first_frame] + length, side="left")
    if idx >= len(distances):
        return -1
    return int(idx)


def load_timing_stats(path: Path) -> tuple[float, float]:
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
    raw = np.loadtxt(path, skiprows=1)
    columns = {name: idx for idx, name in enumerate(header)}
    return float(np.mean(raw[:, columns["TOTAL"]])), float(np.median(raw[:, columns["TOTAL"]]))


def evaluate_run(est_path: Path, gt_path: Path, timing_path: Path) -> dict[str, object]:
    est = np.loadtxt(est_path)
    gt = np.loadtxt(gt_path)
    if est.ndim == 1:
        est = est[None, :]
    if gt.ndim == 1:
        gt = gt[None, :]

    n = min(len(est), len(gt))
    est_poses = [est_row_to_pose(row) for row in est[:n]]
    gt_poses = [gt_row_to_pose(row) for row in gt[:n]]

    pose_correction = gt_poses[0] @ np.linalg.inv(est_poses[0])
    est_first_aligned = [pose_correction @ pose for pose in est_poses]

    rpe1_errors = []
    for i in range(n - 1):
        rel_gt = gt_poses[i][:3, :3].T @ gt_poses[i + 1][:3, :3]
        rel_est = est_first_aligned[i][:3, :3].T @ est_first_aligned[i + 1][:3, :3]
        rpe1_errors.append(rot_angle(rel_gt.T @ rel_est))
    rpe1_deg = np.sqrt(np.mean(np.square(rpe1_errors))) * 180.0 / np.pi

    rpen_values = []
    for distance in range(1, n):
        errors = []
        for i in range(n - distance):
            rel_gt = gt_poses[i][:3, :3].T @ gt_poses[i + distance][:3, :3]
            rel_est = est_first_aligned[i][:3, :3].T @ est_first_aligned[i + distance][:3, :3]
            errors.append(rot_angle(rel_gt.T @ rel_est))
        rpen_values.append(np.sqrt(np.mean(np.square(errors))))
    rpen_deg = np.mean(rpen_values) * 180.0 / np.pi

    est_positions = np.asarray([pose[:3, 3] for pose in est_first_aligned])
    gt_positions = np.asarray([pose[:3, 3] for pose in gt_poses])
    scale, R_align, t_align = umeyama_alignment(est_positions, gt_positions)

    est_sim3 = []
    for pose in est_first_aligned:
        T = np.eye(4)
        T[:3, :3] = R_align @ pose[:3, :3]
        T[:3, 3] = scale * (R_align @ pose[:3, 3]) + t_align
        est_sim3.append(T)

    ate_rmse = np.sqrt(
        np.mean(
            np.sum(
                (
                    np.asarray([pose[:3, 3] for pose in est_sim3])
                    - np.asarray([pose[:3, 3] for pose in gt_poses])
                )
                ** 2,
                axis=1,
            )
        )
    )

    distances = trajectory_distances(gt_poses)
    all_trel = []
    for length in SEGMENT_LENGTHS:
        for first in range(n):
            last = last_frame_from_length(distances, first, length)
            if last == -1:
                continue
            rel_est = np.linalg.inv(est_sim3[first]) @ est_sim3[last]
            rel_gt = np.linalg.inv(gt_poses[first]) @ gt_poses[last]
            rel_err = np.linalg.inv(rel_gt) @ rel_est
            all_trel.append(np.linalg.norm(rel_err[:3, 3]) / float(length))
    trel_pct = float(np.mean(all_trel) * 100.0)

    mean_total_ms, median_total_ms = load_timing_stats(timing_path)

    return {
        "frames": n,
        "RPE1_deg": float(rpe1_deg),
        "RPEn_deg": float(rpen_deg),
        "t_rel_pct": trel_pct,
        "ATE_sim3_m": float(ate_rmse),
        "Sim3_scale": float(scale),
        "mean_total_ms": mean_total_ms,
        "median_total_ms": median_total_ms,
        "gt_positions": gt_positions,
        "est_positions_sim3": np.asarray([pose[:3, 3] for pose in est_sim3]),
    }


def save_plot(path: Path, title: str, gt_positions: np.ndarray, est_positions: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], label="GT", linewidth=2.0)
    ax.plot(est_positions[:, 0], est_positions[:, 2], label=label, linewidth=1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_overlay(path: Path, sequence: str, gt_positions: np.ndarray, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], label="GT", linewidth=2.3, color="black")
    for row in rows:
        est_positions = row["est_positions_sim3"]
        ax.plot(est_positions[:, 0], est_positions[:, 2], linewidth=1.2, label=row["experiment"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"KITTI {sequence} PNEC Experiments vs GT")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate arbitrary KITTI PNEC/NEC experiment folders for one sequence.")
    parser.add_argument("--sequence", required=True, help="KITTI sequence id, e.g. 07.")
    parser.add_argument("--results-root", required=True, help="Root directory containing result folders.")
    parser.add_argument("--gt-root", required=True, help="KITTI sequences root directory.")
    parser.add_argument("--folders", nargs="*", default=None, help="Explicit result folder names.")
    parser.add_argument("--prefix", default=None, help="Only folders matching this prefix are evaluated.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSV and plots.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.folders:
        folder_names = args.folders
    else:
        prefix = args.prefix if args.prefix is not None else f"{args.sequence}_"
        folder_names = sorted(
            [path.name for path in results_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
        )

    gt_path = gt_root / args.sequence / "poses.txt"
    rows = []
    for folder_name in folder_names:
        result_dir = results_root / folder_name
        est_path = result_dir / "rot_avg" / "poses.txt"
        timing_path = result_dir / "timing.txt"
        if not est_path.is_file() or not timing_path.is_file() or not gt_path.is_file():
            print(f"Skipping {folder_name}: missing files")
            continue

        metrics = evaluate_run(est_path, gt_path, timing_path)
        save_plot(
            plot_dir / f"{folder_name}_vs_gt.png",
            f"KITTI {args.sequence} {folder_name} vs GT",
            metrics["gt_positions"],
            metrics["est_positions_sim3"],
            folder_name,
        )
        rows.append(
            {
                "experiment": folder_name,
                "frames": metrics["frames"],
                "RPE1_deg": metrics["RPE1_deg"],
                "RPEn_deg": metrics["RPEn_deg"],
                "t_rel_pct": metrics["t_rel_pct"],
                "ATE_sim3_m": metrics["ATE_sim3_m"],
                "Sim3_scale": metrics["Sim3_scale"],
                "mean_total_ms": metrics["mean_total_ms"],
                "median_total_ms": metrics["median_total_ms"],
                "plot": str(plot_dir / f"{folder_name}_vs_gt.png"),
                "gt_positions": metrics["gt_positions"],
                "est_positions_sim3": metrics["est_positions_sim3"],
            }
        )
        print(
            f"{folder_name}: RPE1={metrics['RPE1_deg']:.6f} deg, "
            f"RPEn={metrics['RPEn_deg']:.6f} deg, "
            f"t_rel={metrics['t_rel_pct']:.6f}%, "
            f"ATE(Sim3)={metrics['ATE_sim3_m']:.6f} m"
        )

    if not rows:
        raise SystemExit("No experiment folders were evaluated.")

    save_overlay(
        plot_dir / f"{args.sequence}_experiments_overlay.png",
        args.sequence,
        rows[0]["gt_positions"],
        rows,
    )

    summary_path = output_dir / f"{args.sequence}_pnec_experiments_summary.csv"
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row.pop("gt_positions")
        csv_row.pop("est_positions_sim3")
        csv_rows.append(csv_row)
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "frames",
                "RPE1_deg",
                "RPEn_deg",
                "t_rel_pct",
                "ATE_sim3_m",
                "Sim3_scale",
                "mean_total_ms",
                "median_total_ms",
                "plot",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"summary_csv={summary_path}")
    print(f"overlay_plot={plot_dir / f'{args.sequence}_experiments_overlay.png'}")


if __name__ == "__main__":
    main()
