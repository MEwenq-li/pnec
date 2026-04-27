import argparse
import csv
from pathlib import Path


FIELDS = [
    "sequence",
    "method",
    "frames",
    "RPE1_deg",
    "RPEn_deg",
    "t_rel_pct",
    "ATE_sim3_m",
    "mean_total_ms",
]


def add_python_rows(rows: list[dict[str, str]], eval_root: Path, sequences: list[str]) -> None:
    for seq in sequences:
        summary_path = eval_root / seq / f"{seq}_pnec_experiments_summary.csv"
        if not summary_path.is_file():
            print(f"Skipping Python summary for {seq}: missing {summary_path}")
            continue
        with open(summary_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "sequence": seq,
                        "method": row["experiment"],
                        "frames": row["frames"],
                        "RPE1_deg": row["RPE1_deg"],
                        "RPEn_deg": row["RPEn_deg"],
                        "t_rel_pct": row["t_rel_pct"],
                        "ATE_sim3_m": row["ATE_sim3_m"],
                        "mean_total_ms": row["mean_total_ms"],
                    }
                )


def add_cpp_rows(rows: list[dict[str, str]], path: Path, label: str, sequences: set[str]) -> None:
    if not path.is_file():
        print(f"Skipping {label}: missing {path}")
        return
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = row["sequence"].zfill(2)
            if seq not in sequences:
                continue
            rows.append(
                {
                    "sequence": seq,
                    "method": label,
                    "frames": row["frames"],
                    "RPE1_deg": row["RPE1_deg"],
                    "RPEn_deg": row["RPEn_deg"],
                    "t_rel_pct": row["t_rel_pct"],
                    "ATE_sim3_m": row["ATE_sim3_m"],
                    "mean_total_ms": row["mean_total_ms"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Python stereo KITTI representative runs.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", nargs="+", default=["00", "03", "05", "07", "09"])
    parser.add_argument("--cpp-nec-summary", default="stereo_nec_eval_all/stereo_nec_kitti_summary.csv")
    parser.add_argument("--cpp-pnec-summary", default="stereo_pnec_eval_all/stereo_pnec_kitti_summary.csv")
    args = parser.parse_args()

    sequences = [seq.zfill(2) for seq in args.sequences]
    rows: list[dict[str, str]] = []
    add_python_rows(rows, Path(args.eval_root), sequences)
    add_cpp_rows(rows, Path(args.cpp_nec_summary), "cpp_stereo_nec", set(sequences))
    add_cpp_rows(rows, Path(args.cpp_pnec_summary), "cpp_stereo_pnec", set(sequences))
    rows.sort(key=lambda row: (row["sequence"], row["method"]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['sequence']} {row['method']} "
            f"RPE1={float(row['RPE1_deg']):.4f} "
            f"RPEn={float(row['RPEn_deg']):.4f} "
            f"t_rel={float(row['t_rel_pct']):.4f} "
            f"ATE={float(row['ATE_sim3_m']):.3f} "
            f"time={float(row['mean_total_ms']):.1f}ms"
        )
    print(f"summary_csv={output}")


if __name__ == "__main__":
    main()
