from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import Pose, angles_from_vec, compose_M, matrix_to_rotvec, normalize, rotvec_to_matrix, skew, translation_from_M, vec_from_angles


@dataclass
class InitialEstimate:
    pose: Pose
    inliers: np.ndarray
    ransac_inliers: int
    pose_inliers: int
    initial_residual_rms: float


def _xy_from_bearings(bvs: np.ndarray) -> np.ndarray:
    bvs = np.asarray(bvs, dtype=np.float64)
    return (bvs[:, :2] / np.maximum(bvs[:, 2:3], 1e-12)).astype(np.float64)


def _essential_mat(host_xy: np.ndarray, target_xy: np.ndarray, threshold: float, max_iters: int):
    try:
        return cv2.findEssentialMat(
            host_xy,
            target_xy,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=threshold,
            maxIters=max_iters,
        )
    except TypeError:
        return cv2.findEssentialMat(
            host_xy,
            target_xy,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=threshold,
        )


def estimate_nec_initial(
    host_bvs: np.ndarray,
    target_bvs: np.ndarray,
    threshold: float = 1e-3,
    max_iters: int = 5000,
    pose_convention: str = "target-to-host",
) -> InitialEstimate:
    if len(host_bvs) < 8:
        inliers = np.arange(len(host_bvs), dtype=np.int32)
        return InitialEstimate(Pose.identity(), inliers, len(inliers), len(inliers), 0.0)
    host_xy = _xy_from_bearings(host_bvs)
    target_xy = _xy_from_bearings(target_bvs)
    E, mask = _essential_mat(host_xy, target_xy, threshold, max_iters)
    if E is None:
        R = np.eye(3, dtype=np.float64)
        t = translation_from_M(compose_M(host_bvs, target_bvs, R))
        inliers = np.arange(len(host_bvs), dtype=np.int32)
        residual = float(np.sqrt(np.mean(_nec_residual(np.array([0.0, 0.0, 0.0, *angles_from_vec(t)]), host_bvs, target_bvs) ** 2)))
        return InitialEstimate(Pose(R, t), inliers, len(inliers), 0, residual)
    if E.ndim == 2 and E.shape[0] > 3:
        E = E[:3, :]
    _, R_ht, t_ht, pose_mask = cv2.recoverPose(E, host_xy, target_xy, focal=1.0, pp=(0.0, 0.0), mask=mask)
    inliers = np.flatnonzero(pose_mask.reshape(-1) > 0).astype(np.int32)
    if len(inliers) == 0:
        inliers = np.flatnonzero(mask.reshape(-1) > 0).astype(np.int32) if mask is not None else np.arange(len(host_bvs), dtype=np.int32)
    if len(inliers) == 0:
        inliers = np.arange(len(host_bvs), dtype=np.int32)

    if pose_convention == "host-to-target":
        R = R_ht.astype(np.float64)
        recover_t = normalize(t_ht.reshape(3).astype(np.float64))
    else:
        R = R_ht.T.astype(np.float64)
        recover_t = normalize((-R_ht.T @ t_ht.reshape(3)).astype(np.float64))

    t = translation_from_M(compose_M(host_bvs[inliers], target_bvs[inliers], R))
    if float(np.dot(t, recover_t)) < 0.0:
        t = -t
    residual = _nec_residual(np.array([*matrix_to_rotvec(R), *angles_from_vec(t)]), host_bvs[inliers], target_bvs[inliers])
    return InitialEstimate(
        Pose(R, t),
        inliers,
        int(np.count_nonzero(mask)) if mask is not None else len(inliers),
        int(np.count_nonzero(pose_mask)) if pose_mask is not None else len(inliers),
        float(np.sqrt(np.mean(residual * residual))) if len(residual) else 0.0,
    )


def _nec_residual(params: np.ndarray, host_bvs: np.ndarray, target_bvs: np.ndarray) -> np.ndarray:
    R = rotvec_to_matrix(params[:3])
    t = vec_from_angles(params[3], params[4])
    rotated_target = target_bvs @ R.T
    return np.cross(host_bvs, rotated_target) @ t


def refine_nec(host_bvs: np.ndarray, target_bvs: np.ndarray, init: Pose, max_nfev: int = 80) -> Pose:
    theta, phi = angles_from_vec(init.t)
    x0 = np.array([*matrix_to_rotvec(init.R), theta, phi], dtype=np.float64)
    result = least_squares(_nec_residual, x0, args=(host_bvs, target_bvs), method="trf", max_nfev=max_nfev)
    return Pose(rotvec_to_matrix(result.x[:3]), vec_from_angles(result.x[3], result.x[4]))


def nec_residual_rms(host_bvs: np.ndarray, target_bvs: np.ndarray, pose: Pose) -> float:
    if len(host_bvs) == 0:
        return 0.0
    theta, phi = angles_from_vec(pose.t)
    residual = _nec_residual(np.array([*matrix_to_rotvec(pose.R), theta, phi], dtype=np.float64), host_bvs, target_bvs)
    return float(np.sqrt(np.mean(residual * residual)))


def _pnec_target_residual(params: np.ndarray, host_bvs: np.ndarray, target_bvs: np.ndarray, target_covs: list[np.ndarray], regularization: float) -> np.ndarray:
    R = rotvec_to_matrix(params[:3])
    t = vec_from_angles(params[3], params[4])
    rotated_target = target_bvs @ R.T
    numerator = np.cross(host_bvs, rotated_target) @ t
    cov2 = np.asarray(target_covs, dtype=np.float64)
    projected = np.cross(t, host_bvs) @ R
    denom = np.einsum("ni,nij,nj->n", projected, cov2, projected) + regularization
    return numerator / np.sqrt(np.maximum(denom, 1e-18))


def _pnec_sym_residual(
    params: np.ndarray,
    host_bvs: np.ndarray,
    target_bvs: np.ndarray,
    host_covs: list[np.ndarray],
    target_covs: list[np.ndarray],
    regularization: float,
) -> np.ndarray:
    R = rotvec_to_matrix(params[:3])
    t = vec_from_angles(params[3], params[4])
    rotated_target = target_bvs @ R.T
    numerator = np.cross(host_bvs, rotated_target) @ t
    cov1 = np.asarray(host_covs, dtype=np.float64)
    cov2 = np.asarray(target_covs, dtype=np.float64)
    target_projected = np.cross(t, host_bvs) @ R
    host_projected = np.cross(t, rotated_target)
    denom = (
        np.einsum("ni,nij,nj->n", target_projected, cov2, target_projected)
        + np.einsum("ni,nij,nj->n", host_projected, cov1, host_projected)
        + regularization
    )
    return numerator / np.sqrt(np.maximum(denom, 1e-18))


def refine_pnec_target(host_bvs: np.ndarray, target_bvs: np.ndarray, target_covs: list[np.ndarray], init: Pose, regularization: float = 1e-10, max_nfev: int = 80) -> Pose:
    theta, phi = angles_from_vec(init.t)
    x0 = np.array([*matrix_to_rotvec(init.R), theta, phi], dtype=np.float64)
    result = least_squares(_pnec_target_residual, x0, args=(host_bvs, target_bvs, target_covs, regularization), method="trf", max_nfev=max_nfev)
    return Pose(rotvec_to_matrix(result.x[:3]), vec_from_angles(result.x[3], result.x[4]))


def refine_pnec_symmetric(
    host_bvs: np.ndarray,
    target_bvs: np.ndarray,
    host_covs: list[np.ndarray],
    target_covs: list[np.ndarray],
    init: Pose,
    regularization: float = 1e-10,
    max_nfev: int = 80,
) -> Pose:
    theta, phi = angles_from_vec(init.t)
    x0 = np.array([*matrix_to_rotvec(init.R), theta, phi], dtype=np.float64)
    result = least_squares(
        _pnec_sym_residual,
        x0,
        args=(host_bvs, target_bvs, host_covs, target_covs, regularization),
        method="trf",
        max_nfev=max_nfev,
    )
    return Pose(rotvec_to_matrix(result.x[:3]), vec_from_angles(result.x[3], result.x[4]))


def metric_translation(R: np.ndarray, host_points_3d: np.ndarray, target_points_3d: np.ndarray, inliers: np.ndarray | None = None) -> np.ndarray:
    if inliers is not None and len(inliers) > 0:
        host_points_3d = host_points_3d[inliers]
        target_points_3d = target_points_3d[inliers]
    if len(host_points_3d) == 0:
        return np.zeros(3, dtype=np.float64)
    candidates = host_points_3d - (R @ target_points_3d.T).T
    median = np.median(candidates, axis=0)
    keep = np.linalg.norm(candidates - median, axis=1) < 3.0
    if not np.any(keep):
        return median
    return np.mean(candidates[keep], axis=0)


def rotation_angle_deg(R: np.ndarray) -> float:
    value = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _kabsch_target_to_host(host_points_3d: np.ndarray, target_points_3d: np.ndarray) -> Pose | None:
    if len(host_points_3d) < 3:
        return None
    host_mean = np.mean(host_points_3d, axis=0)
    target_mean = np.mean(target_points_3d, axis=0)
    host_centered = host_points_3d - host_mean
    target_centered = target_points_3d - target_mean
    H = target_centered.T @ host_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = host_mean - R @ target_mean
    return Pose(R, t)


def rigid_residuals(pose: Pose, host_points_3d: np.ndarray, target_points_3d: np.ndarray) -> np.ndarray:
    predicted_host = (pose.R @ target_points_3d.T).T + pose.t
    return np.linalg.norm(host_points_3d - predicted_host, axis=1)


def estimate_rigid_3d_ransac(
    host_points_3d: np.ndarray,
    target_points_3d: np.ndarray,
    iterations: int = 96,
    threshold_m: float = 1.0,
    min_inliers: int = 12,
) -> tuple[Pose | None, np.ndarray]:
    count = len(host_points_3d)
    if count < 3:
        return None, np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(7)
    best_inliers = np.arange(count, dtype=np.int32)
    best_score = -1
    sample_count = min(iterations, max(iterations, count))
    for _ in range(sample_count):
        sample = rng.choice(count, size=3, replace=False)
        pose = _kabsch_target_to_host(host_points_3d[sample], target_points_3d[sample])
        if pose is None:
            continue
        inliers = np.flatnonzero(rigid_residuals(pose, host_points_3d, target_points_3d) < threshold_m).astype(np.int32)
        if len(inliers) > best_score:
            best_score = len(inliers)
            best_inliers = inliers

    if len(best_inliers) < min_inliers:
        pose = _kabsch_target_to_host(host_points_3d, target_points_3d)
        return pose, np.arange(count, dtype=np.int32)

    pose = _kabsch_target_to_host(host_points_3d[best_inliers], target_points_3d[best_inliers])
    return pose, best_inliers
