import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] == 12:
        # KITTI 3x4 pose matrices per row
        xyz = data[:, [3, 7, 11]]
    elif data.shape[1] == 8:
        # PNEC output: timestamp tx ty tz qx qy qz qw
        xyz = data[:, 1:4]
    else:
        raise ValueError(
            f"Unsupported trajectory format in {path}. Expected 12 or 8 columns, "
            f"got {data.shape[1]}."
        )

    return xyz


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot estimated and GT trajectories.")
    parser.add_argument("--est", required=True, help="Estimated trajectory file.")
    parser.add_argument("--gt", help="Ground-truth trajectory file.")
    parser.add_argument(
        "--output",
        help="Optional output image path. If omitted, an image is saved next to the estimate.",
    )
    parser.add_argument(
        "--title",
        default="Trajectory",
        help="Figure title.",
    )
    args = parser.parse_args()

    est_path = Path(args.est)
    gt_path = Path(args.gt) if args.gt else None

    est_xyz = load_trajectory(est_path)
    gt_xyz = load_trajectory(gt_path) if gt_path else None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(est_xyz[:, 0], est_xyz[:, 2], label="Estimated", linewidth=2.0)

    if gt_xyz is not None:
        limit = min(len(est_xyz), len(gt_xyz))
        ax.plot(gt_xyz[:limit, 0], gt_xyz[:limit, 2], label="Ground Truth", linewidth=2.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(args.title)
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_path = Path(args.output) if args.output else est_path.with_suffix(".png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    main()
