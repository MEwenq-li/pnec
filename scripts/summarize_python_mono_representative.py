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


def add_rows_from_summary(rows: list[dict[str, str]], summary_path: Path, sequence: str, prefix: str = "") -> None:
    if not summary_path.is_file():
        print(f"Skipping missing summary: {summary_path}")
        return
    with open(summary_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = f"{prefix}{row['experiment']}" if prefix else row["experiment"]
            rows.append(
                {
                    "sequence": sequence,
                    "method": method,
                    "frames": row["frames"],
                    "RPE1_deg": row["RPE1_deg"],
                    "RPEn_deg": row["RPEn_deg"],
                    "t_rel_pct": row["t_rel_pct"],
                    "ATE_sim3_m": row["ATE_sim3_m"],
                    "mean_total_ms": row["mean_total_ms"],
                }
            )


def add_cpp_nec_rows(rows: list[dict[str, str]], nec_summary: Path, sequences: set[str]) -> None:
    if not nec_summary.is_file():
        print(f"Skipping missing C++ NEC summary: {nec_summary}")
        return
    with open(nec_summary, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = row["sequence"].zfill(2)
            if seq not in sequences:
                continue
            rows.append(
                {
                    "sequence": seq,
                    "method": "cpp_nec",
                    "frames": row["frames"],
                    "RPE1_deg": row["RPE1_deg"],
                    "RPEn_deg": row["RPEn_deg"],
                    "t_rel_pct": row["t_rel_pct"],
                    "ATE_sim3_m": row["ATE_sim3_m"],
                    "mean_total_ms": row["mean_total_ms"],
                }
            )


def add_available_cpp_pnec_rows(rows: list[dict[str, str]], available_root: Path, sequences: list[str]) -> None:
    if not available_root:
        return
    for seq in sequences:
        summary_path = available_root / seq / f"{seq}_pnec_experiments_summary.csv"
        if not summary_path.is_file():
            continue
        with open(summary_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if "pnec" not in row["experiment"].lower():
                    continue
                rows.append(
                    {
                        "sequence": seq,
                        "method": f"cpp_available_{row['experiment']}",
                        "frames": row["frames"],
                        "RPE1_deg": row["RPE1_deg"],
                        "RPEn_deg": row["RPEn_deg"],
                        "t_rel_pct": row["t_rel_pct"],
                        "ATE_sim3_m": row["ATE_sim3_m"],
                        "mean_total_ms": row["mean_total_ms"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Python mono KITTI runs and optional C++ baselines.")
    parser.add_argument("--eval-root", required=True, help="Root containing per-sequence evaluation directories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", nargs="+", default=["07"])
    parser.add_argument("--cpp-nec-summary", default="nec_eval_all/nec_kitti_summary.csv")
    parser.add_argument("--cpp-pnec-07-summary", default="pnec_eval_07_symmetric/07_pnec_experiments_summary.csv")
    parser.add_argument("--cpp-available-root", default="")
    args = parser.parse_args()

    sequences = [seq.zfill(2) for seq in args.sequences]
    rows: list[dict[str, str]] = []
    for seq in sequences:
        add_rows_from_summary(rows, Path(args.eval_root) / seq / f"{seq}_pnec_experiments_summary.csv", seq)
    add_cpp_nec_rows(rows, Path(args.cpp_nec_summary), set(sequences))
    if args.cpp_available_root:
        add_available_cpp_pnec_rows(rows, Path(args.cpp_available_root), sequences)
    if "07" in set(sequences):
        add_rows_from_summary(rows, Path(args.cpp_pnec_07_summary), "07", "cpp_")

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
