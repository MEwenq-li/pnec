from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import bearing_covariances, pixels_to_bearings
from .io import KittiStereoCalibration


@dataclass
class PairFeatures:
    host_points: np.ndarray
    target_points: np.ndarray
    host_bvs: np.ndarray
    target_bvs: np.ndarray
    host_covs: list[np.ndarray]
    target_covs: list[np.ndarray]
    host_points_3d: np.ndarray | None = None
    target_points_3d: np.ndarray | None = None
    mean_flow_px: float = 0.0


def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect_points(img: np.ndarray, max_corners: int = 2000) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        useHarrisDetector=False,
    )
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def track_points(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    prev_pts: np.ndarray,
    fb_threshold: float = 1.0,
    lk_error_threshold: float = 30.0,
    border: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prev_pts) == 0:
        return prev_pts, prev_pts, np.zeros(0, dtype=bool)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_img,
        curr_img,
        prev_pts.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    curr_pts = curr_pts.reshape(-1, 2)
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        curr_img,
        prev_img,
        curr_pts.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    err = err.reshape(-1)
    status = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(back_pts.reshape(-1, 2) - prev_pts, axis=1)
    status &= fb_error <= fb_threshold
    status &= err <= lk_error_threshold
    h, w = curr_img.shape[:2]
    status &= (curr_pts[:, 0] >= border) & (curr_pts[:, 0] < w - border)
    status &= (curr_pts[:, 1] >= border) & (curr_pts[:, 1] < h - border)
    return prev_pts[status], curr_pts[status], err[status]


def stereo_points(left_img: np.ndarray, right_img: np.ndarray, left_pts: np.ndarray, calib: KittiStereoCalibration, fb_threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(left_pts) == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float64)
    right_init = left_pts.astype(np.float32).reshape(-1, 1, 2)
    right_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        left_img,
        right_img,
        left_pts.astype(np.float32).reshape(-1, 1, 2),
        right_init,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    right_pts = right_pts.reshape(-1, 2)
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        right_img,
        left_img,
        right_pts.astype(np.float32).reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    status = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
    status &= np.linalg.norm(back_pts.reshape(-1, 2) - left_pts, axis=1) <= fb_threshold
    disparity = left_pts[:, 0] - right_pts[:, 0]
    y_error = np.abs(left_pts[:, 1] - right_pts[:, 1])
    depth = calib.fx * calib.baseline / np.maximum(disparity, 1e-12)
    valid = status & (disparity > 0.5) & (y_error <= 1.5) & np.isfinite(depth) & (depth > 0.0) & (depth <= 150.0)
    points_3d = np.zeros((len(left_pts), 3), dtype=np.float64)
    points_3d[:, 0] = (left_pts[:, 0] - calib.cx) * depth / calib.fx
    points_3d[:, 1] = (left_pts[:, 1] - calib.cy) * depth / calib.fy
    points_3d[:, 2] = depth
    return valid, right_pts, points_3d


def build_pair_features(
    host_left: np.ndarray,
    target_left: np.ndarray,
    calib: KittiStereoCalibration,
    host_right: np.ndarray | None = None,
    target_right: np.ndarray | None = None,
    max_corners: int = 1200,
    pixel_sigma: float = 1.0,
) -> PairFeatures:
    host_seed = detect_points(host_left, max_corners=max_corners)
    host_pts, target_pts, _ = track_points(host_left, target_left, host_seed)

    if host_right is not None and target_right is not None:
        host_valid, _, host_3d = stereo_points(host_left, host_right, host_pts, calib)
        target_valid, _, target_3d = stereo_points(target_left, target_right, target_pts, calib)
        valid = host_valid & target_valid
        host_pts = host_pts[valid]
        target_pts = target_pts[valid]
        host_3d = host_3d[valid]
        target_3d = target_3d[valid]
    else:
        host_3d = None
        target_3d = None

    if len(host_pts) > 0:
        mean_flow_px = float(np.mean(np.linalg.norm(target_pts - host_pts, axis=1)))
    else:
        mean_flow_px = 0.0

    host_bvs = pixels_to_bearings(host_pts, calib.fx, calib.fy, calib.cx, calib.cy)
    target_bvs = pixels_to_bearings(target_pts, calib.fx, calib.fy, calib.cx, calib.cy)
    host_covs = bearing_covariances(host_pts, calib.fx, calib.fy, calib.cx, calib.cy, pixel_sigma)
    target_covs = bearing_covariances(target_pts, calib.fx, calib.fy, calib.cx, calib.cy, pixel_sigma)
    return PairFeatures(host_pts, target_pts, host_bvs, target_bvs, host_covs, target_covs, host_3d, target_3d, mean_flow_px)
