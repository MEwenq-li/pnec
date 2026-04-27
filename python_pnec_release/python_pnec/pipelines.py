from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frontend import build_pair_features, read_gray
from .geometry import Pose
from .io import list_images, load_kitti_stereo_calibration, load_timestamps, prepare_output, sequence_paths, write_poses, write_timing
from .solvers import (
    estimate_nec_initial,
    estimate_rigid_3d_ransac,
    metric_translation,
    nec_residual_rms,
    refine_nec,
    refine_pnec_symmetric,
    refine_pnec_target,
    rigid_residuals,
    rotation_angle_deg,
)


@dataclass
class SolveResult:
    pose: Pose | None
    inliers: np.ndarray
    ransac_inliers: int = 0
    pose_inliers: int = 0
    initial_residual_rms: float = 0.0
    final_residual_rms: float = 0.0
    rel_rotation_deg: float = 0.0
    rel_translation_norm: float = 0.0


def _solve_pair(
    features,
    method: str,
    min_matches: int,
    regularization: float,
    ransac_threshold: float,
    ransac_iters: int,
    pose_convention: str,
) -> SolveResult:
    if len(features.host_bvs) < min_matches:
        return SolveResult(None, np.zeros(0, dtype=np.int32))
    initial = estimate_nec_initial(
        features.host_bvs,
        features.target_bvs,
        threshold=ransac_threshold,
        max_iters=ransac_iters,
        pose_convention=pose_convention,
    )
    inliers = initial.inliers
    if len(inliers) >= min_matches:
        hb = features.host_bvs[inliers]
        tb = features.target_bvs[inliers]
        hc = [features.host_covs[int(i)] for i in inliers]
        tc = [features.target_covs[int(i)] for i in inliers]
    else:
        hb = features.host_bvs
        tb = features.target_bvs
        hc = features.host_covs
        tc = features.target_covs
        inliers = np.arange(len(hb), dtype=np.int32)

    nec = refine_nec(hb, tb, initial.pose)
    if method == "nec":
        pose = nec
    if method == "pnec_target":
        pose = refine_pnec_target(hb, tb, tc, nec, regularization=regularization)
    elif method != "nec":
        pose = refine_pnec_symmetric(hb, tb, hc, tc, nec, regularization=regularization)

    return SolveResult(
        pose,
        inliers,
        ransac_inliers=initial.ransac_inliers,
        pose_inliers=initial.pose_inliers,
        initial_residual_rms=initial.initial_residual_rms,
        final_residual_rms=nec_residual_rms(hb, tb, pose),
        rel_rotation_deg=rotation_angle_deg(pose.R),
        rel_translation_norm=float(np.linalg.norm(pose.t)),
    )


def _write_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sequence(
    dataset_root: Path,
    sequence: str,
    output_dir: Path,
    method: str,
    stereo: bool,
    max_corners: int = 2000,
    min_matches: int = 20,
    regularization: float = 1e-10,
    max_frames: int | None = None,
    ransac_threshold: float = 1e-3,
    ransac_iters: int = 5000,
    diagnostics: bool = False,
    min_local_rad: float = 5.0,
    max_mono_rotation_deg: float = 5.0,
    pose_convention: str = "target-to-host",
) -> None:
    left_dir, right_dir, times_path, calib_path, _ = sequence_paths(Path(dataset_root), sequence)
    left_images = list_images(left_dir)
    timestamps = load_timestamps(times_path)
    calib = load_kitti_stereo_calibration(calib_path)
    right_images = list_images(right_dir) if stereo else []

    pose_path = prepare_output(output_dir)
    poses: list[tuple[float, Pose]] = []
    timing_rows: list[dict[str, float]] = []
    diagnostic_rows: list[dict[str, object]] = []

    global_pose = Pose.identity()
    prev_rel = Pose.identity()
    poses.append((float(timestamps[0]), global_pose))
    prev_left = read_gray(left_images[0])
    prev_right = read_gray(right_images[0]) if stereo else None
    timing_rows.append({"ID": 0, "FrameLoading": 0, "FeatureCreation": 0, "NEC-ES": 0, "IT-ES": 0, "AVG-IT-ES": 0, "CERES": 0, "OPTIMIZATION": 0, "TOTAL": 0})

    frame_limit = min(len(left_images), len(timestamps))
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames)

    for idx in range(1, frame_limit):
        frame_start = time.perf_counter()
        load_start = time.perf_counter()
        curr_left = read_gray(left_images[idx])
        curr_right = read_gray(right_images[idx]) if stereo else None
        frame_loading_ms = (time.perf_counter() - load_start) * 1000.0

        feature_start = time.perf_counter()
        features = build_pair_features(
            prev_left,
            curr_left,
            calib,
            prev_right,
            curr_right,
            max_corners=max_corners,
        )
        feature_ms = (time.perf_counter() - feature_start) * 1000.0

        opt_start = time.perf_counter()
        skipped = (not stereo) and features.mean_flow_px < min_local_rad
        solve = _solve_pair(
            features,
            method,
            min_matches,
            regularization,
            ransac_threshold,
            ransac_iters,
            pose_convention,
        )
        opt_ms = (time.perf_counter() - opt_start) * 1000.0
        rel_pose = solve.pose
        if (not stereo) and solve.rel_rotation_deg > max_mono_rotation_deg:
            skipped = True
        if rel_pose is not None and not skipped:
            if stereo and features.host_points_3d is not None and features.target_points_3d is not None:
                t_metric = metric_translation(rel_pose.R, features.host_points_3d, features.target_points_3d, solve.inliers)
                rel_pose = Pose(rel_pose.R, t_metric)
                rigid_pose, _ = estimate_rigid_3d_ransac(features.host_points_3d, features.target_points_3d)
                if rigid_pose is not None:
                    candidate_residual = np.median(rigid_residuals(rel_pose, features.host_points_3d, features.target_points_3d))
                    rigid_residual = np.median(rigid_residuals(rigid_pose, features.host_points_3d, features.target_points_3d))
                    is_gross_rotation = rotation_angle_deg(rel_pose.R) > 15.0
                    is_gross_translation = np.linalg.norm(rel_pose.t) > 10.0
                    is_poor_3d_fit = candidate_residual > max(2.5 * rigid_residual, rigid_residual + 2.0)
                    if is_gross_rotation or is_gross_translation or is_poor_3d_fit:
                        rel_pose = rigid_pose
            global_pose = global_pose @ rel_pose
            prev_rel = rel_pose
        poses.append((float(timestamps[idx]), global_pose))

        if diagnostics:
            diagnostic_rows.append(
                {
                    "ID": idx,
                    "matches": len(features.host_bvs),
                    "mean_flow_px": features.mean_flow_px,
                    "ransac_inliers": solve.ransac_inliers,
                    "pose_inliers": solve.pose_inliers,
                    "used_inliers": len(solve.inliers),
                    "initial_residual_rms": solve.initial_residual_rms,
                    "final_residual_rms": solve.final_residual_rms,
                    "rel_rotation_deg": solve.rel_rotation_deg,
                    "rel_translation_norm": solve.rel_translation_norm,
                    "skipped": int(skipped),
                    "gross_rotation": int(solve.rel_rotation_deg > 15.0),
                    "rotation_rejected": int((not stereo) and solve.rel_rotation_deg > max_mono_rotation_deg),
                    "pose_convention": pose_convention,
                }
            )

        total_ms = (time.perf_counter() - frame_start) * 1000.0
        timing_rows.append(
            {
                "ID": idx,
                "FrameLoading": frame_loading_ms,
                "FeatureCreation": feature_ms,
                "NEC-ES": 0,
                "IT-ES": 0,
                "AVG-IT-ES": 0,
                "CERES": opt_ms,
                "OPTIMIZATION": opt_ms,
                "TOTAL": total_ms,
            }
        )
        prev_left = curr_left
        prev_right = curr_right

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_poses(pose_path, poses)
    write_timing(output_dir / "timing.txt", timing_rows)
    if diagnostics:
        _write_diagnostics(output_dir / "diagnostics.csv", diagnostic_rows)
