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


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def load_estimated_poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 8:
        raise ValueError(f"Expected estimated trajectory with 8 columns, got {data.shape[1]}")
    return data


def load_gt_poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 12:
        raise ValueError(f"Expected ground truth trajectory with 12 columns, got {data.shape[1]}")
    return data


def est_row_to_pose(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = row[1:4]
    R = quat_to_rot(row[4], row[5], row[6], row[7])
    return R, t


def gt_row_to_pose(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = row.reshape(3, 4)
    return T[:, :3], T[:, 3]


def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # Maps X to Y with Sim(3): Y ~= s * R * X + t
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a monocular trajectory to GT with Umeyama Sim(3).")
    parser.add_argument("--est", required=True, help="Estimated trajectory file (timestamp tx ty tz qx qy qz qw).")
    parser.add_argument("--gt", required=True, help="Ground truth trajectory file (KITTI 3x4 per row).")
    parser.add_argument("--output-est", required=True, help="Output aligned estimated trajectory file.")
    parser.add_argument("--output-plot", required=True, help="Output plot image path.")
    parser.add_argument("--title", default="Sim(3) Aligned Trajectory", help="Plot title.")
    args = parser.parse_args()

    est_path = Path(args.est)
    gt_path = Path(args.gt)
    out_est = Path(args.output_est)
    out_plot = Path(args.output_plot)

    est = load_estimated_poses(est_path)
    gt = load_gt_poses(gt_path)
    n = min(len(est), len(gt))
    est = est[:n]
    gt = gt[:n]

    est_positions = est[:, 1:4]
    gt_positions = np.asarray([gt_row_to_pose(row)[1] for row in gt])

    scale, R_align, t_align = umeyama_alignment(est_positions, gt_positions)

    aligned = est.copy()
    for i in range(n):
        R_est, t_est = est_row_to_pose(est[i])
        R_new = R_align @ R_est
        t_new = scale * (R_align @ t_est) + t_align
        q_new = rot_to_quat(R_new)
        aligned[i, 1:4] = t_new
        aligned[i, 4:8] = q_new

    out_est.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_est,
        aligned,
        fmt="%.6f %.8e %.8e %.8e %.8e %.8e %.8e %.8e",
    )

    aligned_positions = aligned[:, 1:4]
    ate_rmse = np.sqrt(np.mean(np.sum((aligned_positions - gt_positions) ** 2, axis=1)))

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label="GT", linewidth=2.0)
    plt.plot(aligned_positions[:, 0], aligned_positions[:, 2], label="Estimate (Sim3 aligned)", linewidth=1.5)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(args.title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot)

    print(f"matched_frames={n}")
    print(f"scale={scale:.8f}")
    print(f"ate_sim3_rmse_m={ate_rmse:.6f}")
    print(f"aligned_est={out_est}")
    print(f"plot={out_plot}")


if __name__ == "__main__":
    main()
