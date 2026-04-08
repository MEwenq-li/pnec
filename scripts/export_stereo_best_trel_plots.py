import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEQUENCES = [f"{i:02d}" for i in range(11)]
COLOR_GT = "#111111"
COLOR_BEST = "#F47720"


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
        aligned.append(scale * (R_align @ pose[:3, 3]) + t_align)
    return np.asarray(aligned)


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["sequence"]: row for row in reader}


def save_plot(output_path: Path, title: str, gt_xyz: np.ndarray, best_xyz: np.ndarray, best_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=220)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color=COLOR_GT, linewidth=2.8, linestyle="-", label="GT")
    ax.plot(best_xyz[:, 0], best_xyz[:, 2], color=COLOR_BEST, linewidth=2.3, linestyle=(0, (7, 2.2)), label=best_label)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one best stereo trajectory plot per KITTI sequence by t_rel.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--nec-summary", required=True)
    parser.add_argument("--pnec-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nec_summary = load_summary(Path(args.nec_summary))
    pnec_summary = load_summary(Path(args.pnec_summary))

    rows = []
    for seq in SEQUENCES:
        nec_row = nec_summary.get(seq)
        pnec_row = pnec_summary.get(seq)
        if nec_row is None or pnec_row is None:
            continue

        use_pnec = float(pnec_row["t_rel_pct"]) <= float(nec_row["t_rel_pct"])
        best_method = "Stereo PNEC" if use_pnec else "Stereo NEC"
        best_dir = f"{seq}_stereo_pnec" if use_pnec else f"{seq}_stereo_nec"
        best_row = pnec_row if use_pnec else nec_row

        gt_poses = load_gt_poses(gt_root / seq / "poses.txt")
        gt_xyz = np.asarray([pose[:3, 3] for pose in gt_poses])
        best_xyz = align_positions(
            load_estimated_poses(results_root / best_dir / "rot_avg" / "poses.txt"),
            gt_poses,
        )

        output_path = output_dir / f"{seq}_best_trel.png"
        save_plot(output_path, f"KITTI {seq}: Best Stereo Trajectory by t_rel", gt_xyz, best_xyz, best_method)

        rows.append(
            {
                "sequence": seq,
                "best_method": best_method,
                "best_t_rel_pct": best_row["t_rel_pct"],
                "best_RPE1_deg": best_row["RPE1_deg"],
                "best_RPEn_deg": best_row["RPEn_deg"],
                "plot": str(output_path),
            }
        )

    summary_path = output_dir / "best_trel_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "best_method",
                "best_t_rel_pct",
                "best_RPE1_deg",
                "best_RPEn_deg",
                "plot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"best_summary={summary_path}")


if __name__ == "__main__":
    main()
