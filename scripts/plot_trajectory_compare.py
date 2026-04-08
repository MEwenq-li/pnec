import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def align_estimate(est_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> list[np.ndarray]:
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
    return est_sim3


def parse_est_arg(arg: str) -> tuple[str, Path]:
    label, value = arg.split("=", 1)
    return label, Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT with one or more estimated trajectories after Sim(3) alignment.")
    parser.add_argument("--gt", required=True, help="KITTI ground-truth trajectory (3x4 per row).")
    parser.add_argument(
        "--est",
        required=True,
        action="append",
        help="Estimated trajectory in form label=path. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--title", default="Trajectory Comparison", help="Plot title.")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    gt_poses = load_gt_poses(gt_path)
    gt_xyz = np.asarray([pose[:3, 3] for pose in gt_poses])

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2.2)

    for est_arg in args.est:
        label, est_path = parse_est_arg(est_arg)
        est_poses = load_estimated_poses(est_path)
        est_aligned = align_estimate(est_poses, gt_poses)
        est_xyz = np.asarray([pose[:3, 3] for pose in est_aligned])
        plt.plot(est_xyz[:, 0], est_xyz[:, 2], label=label, linewidth=1.4)

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(args.title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
