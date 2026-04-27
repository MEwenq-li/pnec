import argparse
from pathlib import Path

from python_pnec.pipelines import run_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure Python stereo NEC.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    out = Path(args.output_root) / f"{args.sequence}_py_stereo_nec"
    run_sequence(Path(args.dataset_root), args.sequence, out, method="nec", stereo=True, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
