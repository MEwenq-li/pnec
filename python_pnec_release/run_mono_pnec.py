import argparse
from pathlib import Path

from python_pnec.pipelines import run_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure Python monocular PNEC.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=["target", "symmetric"], default="symmetric")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-corners", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=1e-2)
    parser.add_argument("--ransac-iters", type=int, default=5000)
    parser.add_argument("--min-local-rad", type=float, default=5.0)
    parser.add_argument("--max-mono-rotation-deg", type=float, default=5.0)
    parser.add_argument("--pose-convention", choices=["target-to-host", "host-to-target"], default="target-to-host")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_root) / f"{args.sequence}_py_mono_pnec_{args.variant}"
    method = "pnec_target" if args.variant == "target" else "pnec_symmetric"
    run_sequence(
        Path(args.dataset_root),
        args.sequence,
        out,
        method=method,
        stereo=False,
        max_frames=args.max_frames,
        max_corners=args.max_corners,
        ransac_threshold=args.ransac_threshold,
        ransac_iters=args.ransac_iters,
        min_local_rad=args.min_local_rad,
        max_mono_rotation_deg=args.max_mono_rotation_deg,
        pose_convention=args.pose_convention,
        diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()
