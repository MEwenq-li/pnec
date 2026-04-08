import argparse
from pathlib import Path

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


def load_estimated(path: Path) -> list[np.ndarray]:
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


def load_gt(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, :4] = row.reshape(3, 4)
        poses.append(T)
    return poses


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


def sim3_align(est_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    n = min(len(est_poses), len(gt_poses))
    est_poses = est_poses[:n]
    gt_poses = gt_poses[:n]

    corr = gt_poses[0] @ np.linalg.inv(est_poses[0])
    est_aligned = [corr @ T for T in est_poses]

    P_est = np.asarray([T[:3, 3] for T in est_aligned])
    P_gt = np.asarray([T[:3, 3] for T in gt_poses])
    scale, R_align, t_align = umeyama_alignment(P_est, P_gt)

    aligned = []
    for T in est_aligned:
        T_new = np.eye(4)
        T_new[:3, :3] = R_align @ T[:3, :3]
        T_new[:3, 3] = scale * (R_align @ T[:3, 3]) + t_align
        aligned.append(T_new)
    return aligned, scale


def trajectory_distances(poses: list[np.ndarray]) -> np.ndarray:
    distances = [0.0]
    for i in range(1, len(poses)):
        p1 = poses[i - 1][:3, 3]
        p2 = poses[i][:3, 3]
        distances.append(distances[-1] + np.linalg.norm(p2 - p1))
    return np.asarray(distances)


def last_frame_from_length(distances: np.ndarray, first_frame: int, length: float) -> int:
    target_distance = distances[first_frame] + length
    idx = np.searchsorted(distances, target_distance, side="left")
    if idx >= len(distances):
        return -1
    return int(idx)


def segment_trel(est_poses: list[np.ndarray], gt_poses: list[np.ndarray], lengths: list[int]) -> dict[int, dict[str, float]]:
    n = min(len(est_poses), len(gt_poses))
    est_poses = est_poses[:n]
    gt_poses = gt_poses[:n]
    distances = trajectory_distances(gt_poses)

    results = {}
    for length in lengths:
        errs = []
        for first in range(n):
            last = last_frame_from_length(distances, first, length)
            if last == -1:
                continue
            rel_est = np.linalg.inv(est_poses[first]) @ est_poses[last]
            rel_gt = np.linalg.inv(gt_poses[first]) @ gt_poses[last]
            err = np.linalg.inv(rel_gt) @ rel_est
            trans_err = np.linalg.norm(err[:3, 3]) / float(length)
            errs.append(trans_err)

        if errs:
            errs = np.asarray(errs)
            results[length] = {
                "mean": float(np.mean(errs)),
                "rmse": float(np.sqrt(np.mean(errs ** 2))),
                "count": int(len(errs)),
            }
        else:
            results[length] = {"mean": float("nan"), "rmse": float("nan"), "count": 0}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute KITTI fixed-length translational relative error.")
    parser.add_argument("--est", required=True, help="Estimated poses file (timestamp tx ty tz qx qy qz qw).")
    parser.add_argument("--gt", required=True, help="GT poses file (KITTI 3x4).")
    parser.add_argument("--no-align", action="store_true", help="Disable global Sim(3) alignment before evaluation.")
    args = parser.parse_args()

    est = load_estimated(Path(args.est))
    gt = load_gt(Path(args.gt))

    scale = None
    if args.no_align:
        est_eval = est
    else:
        est_eval, scale = sim3_align(est, gt)

    results = segment_trel(est_eval, gt, SEGMENT_LENGTHS)

    if scale is not None:
        print(f"sim3_scale={scale:.8f}")
    for length in SEGMENT_LENGTHS:
        r = results[length]
        print(
            f"{length}m: mean_trel={r['mean'] * 100:.4f}% "
            f"rmse_trel={r['rmse'] * 100:.4f}% count={r['count']}"
        )


if __name__ == "__main__":
    main()
