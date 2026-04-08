import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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


def est_row_to_pose(row: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(row[4], row[5], row[6], row[7])
    T[:3, 3] = row[1:4]
    return T


def gt_row_to_pose(row: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = row.reshape(3, 4)
    return T


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


def load_estimated_poses(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return [est_row_to_pose(row) for row in data]


def load_gt_poses(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return [gt_row_to_pose(row) for row in data]


def align_estimate(est_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> np.ndarray:
    n = min(len(est_poses), len(gt_poses))
    est_poses = est_poses[:n]
    gt_poses = gt_poses[:n]

    first_pose_correction = gt_poses[0] @ np.linalg.inv(est_poses[0])
    est_aligned = [first_pose_correction @ pose for pose in est_poses]

    est_positions = np.asarray([pose[:3, 3] for pose in est_aligned])
    gt_positions = np.asarray([pose[:3, 3] for pose in gt_poses])
    scale, R_align, t_align = umeyama_alignment(est_positions, gt_positions)

    est_sim3 = []
    for pose in est_aligned:
        T = np.eye(4)
        T[:3, :3] = R_align @ pose[:3, :3]
        T[:3, 3] = scale * (R_align @ pose[:3, 3]) + t_align
        est_sim3.append(T)
    return np.asarray([pose[:3, 3] for pose in est_sim3])


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["sequence"]: row for row in reader}


def plot_compare(output_path: Path, title: str, gt_xyz: np.ndarray, nec_xyz: np.ndarray, pnec_xyz: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2.2)
    ax.plot(nec_xyz[:, 0], nec_xyz[:, 2], label="Stereo NEC", linewidth=1.4)
    ax.plot(pnec_xyz[:, 0], pnec_xyz[:, 2], label="Stereo PNEC", linewidth=1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KITTI stereo NEC and stereo PNEC trajectories.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--nec-summary", required=True)
    parser.add_argument("--pnec-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    nec_summary = load_summary(Path(args.nec_summary))
    pnec_summary = load_summary(Path(args.pnec_summary))

    rows = []
    for seq in SEQUENCES:
        nec_row = nec_summary.get(seq)
        pnec_row = pnec_summary.get(seq)
        if nec_row is None or pnec_row is None:
            continue

        gt_poses = load_gt_poses(gt_root / seq / "poses.txt")
        gt_xyz = np.asarray([pose[:3, 3] for pose in gt_poses])

        nec_est = load_estimated_poses(results_root / f"{seq}_stereo_nec" / "rot_avg" / "poses.txt")
        pnec_est = load_estimated_poses(results_root / f"{seq}_stereo_pnec" / "rot_avg" / "poses.txt")
        nec_xyz = align_estimate(nec_est, gt_poses)
        pnec_xyz = align_estimate(pnec_est, gt_poses)

        compare_plot = plot_dir / f"{seq}_stereo_pnec_nec_gt.png"
        plot_compare(compare_plot, f"KITTI {seq} Stereo PNEC / NEC / GT", gt_xyz, nec_xyz, pnec_xyz)

        rows.append(
            {
                "sequence": seq,
                "RPE1_deg_nec": nec_row["RPE1_deg"],
                "RPEn_deg_nec": nec_row["RPEn_deg"],
                "t_rel_pct_nec": nec_row["t_rel_pct"],
                "RPE1_deg_pnec": pnec_row["RPE1_deg"],
                "RPEn_deg_pnec": pnec_row["RPEn_deg"],
                "t_rel_pct_pnec": pnec_row["t_rel_pct"],
                "mean_total_ms_nec": nec_row["mean_total_ms"],
                "mean_total_ms_pnec": pnec_row["mean_total_ms"],
                "pnec_vs_gt_plot": pnec_row["plot"],
                "pnec_nec_gt_plot": str(compare_plot),
            }
        )

    summary_path = output_dir / "stereo_nec_pnec_comparison.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "RPE1_deg_nec",
                "RPEn_deg_nec",
                "t_rel_pct_nec",
                "RPE1_deg_pnec",
                "RPEn_deg_pnec",
                "t_rel_pct_pnec",
                "mean_total_ms_nec",
                "mean_total_ms_pnec",
                "pnec_vs_gt_plot",
                "pnec_nec_gt_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"comparison_csv={summary_path}")


if __name__ == "__main__":
    main()
