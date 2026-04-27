from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.R, self.R @ other.t + self.t)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def pixels_to_bearings(points: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xy = np.column_stack(((points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy))
    return normalize(np.column_stack((xy, np.ones(len(xy), dtype=np.float64))))


def bearing_covariances(points: np.ndarray, fx: float, fy: float, cx: float, cy: float, pixel_sigma: float = 1.0) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    sigma_u = pixel_sigma / max(float(fx), 1e-12)
    sigma_v = pixel_sigma / max(float(fy), 1e-12)
    cov2 = np.diag([sigma_u * sigma_u, sigma_v * sigma_v])
    covs: list[np.ndarray] = []
    for point in points:
        xy = np.array([(point[0] - cx) / fx, (point[1] - cy) / fy], dtype=np.float64)
        q = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            covs.append(np.eye(3, dtype=np.float64) * 1e-12)
            continue
        b = q / q_norm
        j_full = (np.eye(3, dtype=np.float64) - np.outer(b, b)) / q_norm
        j = j_full[:, :2]
        covs.append(j @ cov2 @ j.T + np.eye(3, dtype=np.float64) * 1e-12)
    return covs


def angles_from_vec(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 0.0, 0.0
    v = v / n
    theta = float(np.arccos(np.clip(v[2], -1.0, 1.0)))
    phi = 0.0 if abs(theta) < 1e-10 else float(np.arctan2(v[1], v[0]))
    return theta, phi


def vec_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def translation_from_M(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((M + M.T) * 0.5)
    return normalize(vecs[:, int(np.argmin(vals))])


def compose_M(bvs1: np.ndarray, bvs2: np.ndarray, R: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.float64)
    for f1, f2 in zip(bvs1, bvs2):
        n = np.cross(f1, R @ f2)
        M += np.outer(n, n)
    return M


def pose_row(timestamp: float, pose: Pose) -> str:
    qx, qy, qz, qw = rot_to_quat_xyzw(pose.R)
    tx, ty, tz = pose.t
    return f"{timestamp:.9f} {tx:.12g} {ty:.12g} {tz:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}\n"

