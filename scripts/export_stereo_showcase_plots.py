import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEQUENCES = ["00", "03", "05", "09"]
COLOR_BEST = "#F47720"
COLOR_WORSE = "#3458A7"
COLOR_GT = "#111111"


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


def align_positions(est_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> np.ndarray:
    n = min(len(est_poses), len(gt_poses))
    est_poses = est_poses[:n]
    gt_poses = gt_poses[:n]

    first_pose_correction = gt_poses[0] @ np.linalg.inv(est_poses[0])
    est_aligned = [first_pose_correction @ pose for pose in est_poses]

    est_positions = np.asarray([pose[:3, 3] for pose in est_aligned])
    gt_positions = np.asarray([pose[:3, 3] for pose in gt_poses])
    scale, R_align, t_align = umeyama_alignment(est_positions, gt_positions)

    aligned = []
    for pose in est_aligned:
        point = scale * (R_align @ pose[:3, 3]) + t_align
        aligned.append(point)
    return np.asarray(aligned)


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["sequence"]: row for row in reader}


def winner(row: dict[str, str]) -> str:
    nec_wins = 0
    pnec_wins = 0
    for metric_nec, metric_pnec in [
        ("RPE1_deg_nec", "RPE1_deg_pnec"),
        ("RPEn_deg_nec", "RPEn_deg_pnec"),
        ("t_rel_pct_nec", "t_rel_pct_pnec"),
    ]:
        if float(row[metric_pnec]) < float(row[metric_nec]):
            pnec_wins += 1
        elif float(row[metric_pnec]) > float(row[metric_nec]):
            nec_wins += 1
    return "pnec" if pnec_wins >= nec_wins else "nec"


def base_axes_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.4)


def save_pnec_gt(output_path: Path, title: str, gt_xyz: np.ndarray, pnec_xyz: np.ndarray, pnec_color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=220)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color=COLOR_GT, linewidth=2.6, linestyle="-", label="GT")
    ax.plot(pnec_xyz[:, 0], pnec_xyz[:, 2], color=pnec_color, linewidth=2.2, linestyle=(0, (6, 2)), label="Stereo PNEC")
    base_axes_style(ax, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_pnec_nec_gt(
    output_path: Path,
    title: str,
    gt_xyz: np.ndarray,
    pnec_xyz: np.ndarray,
    nec_xyz: np.ndarray,
    winner_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=220)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color=COLOR_GT, linewidth=2.7, linestyle="-", label="GT")

    if winner_name == "pnec":
        ax.plot(pnec_xyz[:, 0], pnec_xyz[:, 2], color=COLOR_BEST, linewidth=2.4, linestyle=(0, (7, 2)), label="Stereo PNEC")
        ax.plot(nec_xyz[:, 0], nec_xyz[:, 2], color=COLOR_WORSE, linewidth=2.0, linestyle=(0, (2, 2)), label="Stereo NEC")
    else:
        ax.plot(nec_xyz[:, 0], nec_xyz[:, 2], color=COLOR_BEST, linewidth=2.4, linestyle=(0, (7, 2)), label="Stereo NEC")
        ax.plot(pnec_xyz[:, 0], pnec_xyz[:, 2], color=COLOR_WORSE, linewidth=2.0, linestyle=(0, (2, 2)), label="Stereo PNEC")

    base_axes_style(ax, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export curated stereo NEC/PNEC comparison plots.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequences", nargs="*", default=DEFAULT_SEQUENCES)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_summary(Path(args.comparison_csv))
    summary_rows = []

    for seq in args.sequences:
        row = comparison[seq]
        win = winner(row)

        gt_poses = load_gt_poses(gt_root / seq / "poses.txt")
        gt_xyz = np.asarray([pose[:3, 3] for pose in gt_poses])

        pnec_xyz = align_positions(
            load_estimated_poses(results_root / f"{seq}_stereo_pnec" / "rot_avg" / "poses.txt"),
            gt_poses,
        )
        nec_xyz = align_positions(
            load_estimated_poses(results_root / f"{seq}_stereo_nec" / "rot_avg" / "poses.txt"),
            gt_poses,
        )

        pnec_color = COLOR_BEST if win == "pnec" else COLOR_WORSE
        pnec_gt_path = output_dir / f"{seq}_pnec_vs_gt_showcase.png"
        full_compare_path = output_dir / f"{seq}_pnec_nec_gt_showcase.png"

        save_pnec_gt(pnec_gt_path, f"KITTI {seq}: Stereo PNEC vs GT", gt_xyz, pnec_xyz, pnec_color)
        save_pnec_nec_gt(full_compare_path, f"KITTI {seq}: Stereo PNEC / NEC / GT", gt_xyz, pnec_xyz, nec_xyz, win)

        summary_rows.append(
            {
                "sequence": seq,
                "winner": win,
                "RPE1_deg_nec": row["RPE1_deg_nec"],
                "RPEn_deg_nec": row["RPEn_deg_nec"],
                "t_rel_pct_nec": row["t_rel_pct_nec"],
                "RPE1_deg_pnec": row["RPE1_deg_pnec"],
                "RPEn_deg_pnec": row["RPEn_deg_pnec"],
                "t_rel_pct_pnec": row["t_rel_pct_pnec"],
                "pnec_vs_gt_plot": str(pnec_gt_path),
                "pnec_nec_gt_plot": str(full_compare_path),
            }
        )

    summary_path = output_dir / "showcase_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "winner",
                "RPE1_deg_nec",
                "RPEn_deg_nec",
                "t_rel_pct_nec",
                "RPE1_deg_pnec",
                "RPEn_deg_pnec",
                "t_rel_pct_pnec",
                "pnec_vs_gt_plot",
                "pnec_nec_gt_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"showcase_summary={summary_path}")


if __name__ == "__main__":
    main()
