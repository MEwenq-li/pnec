import argparse
import csv
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize covariance_stats.csv exported by PNEC diagnostics.")
    parser.add_argument("--input", required=True, help="Path to covariance_stats.csv")
    args = parser.parse_args()

    path = Path(args.input)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))

    fields = [
        "host_mean_trace",
        "host_mean_min_eig",
        "host_mean_max_eig",
        "host_mean_condition",
        "target_mean_trace",
        "target_mean_min_eig",
        "target_mean_max_eig",
        "target_mean_condition",
        "projected_mean_trace",
        "projected_mean_min_eig",
        "projected_mean_max_eig",
        "projected_mean_condition",
    ]

    values = {field: [float(row[field]) for row in rows] for field in fields}

    print(f"rows={len(rows)}")
    for field in fields:
        series = values[field]
        print(
            f"{field}: "
            f"mean={sum(series) / len(series):.12e} "
            f"median={quantile(series, 0.5):.12e} "
            f"p90={quantile(series, 0.9):.12e} "
            f"max={max(series):.12e}"
        )

    projected_min = values["projected_mean_min_eig"]
    projected_trace = values["projected_mean_trace"]
    zeroish_ratio = sum(1 for value in projected_min if value <= 1.0e-12) / len(projected_min)
    spike_ratio = sum(1 for value in projected_trace if value > 1.0e-4) / len(projected_trace)

    print(f"projected_zeroish_min_eig_ratio={zeroish_ratio:.6f}")
    print(f"projected_trace_gt_1e-4_ratio={spike_ratio:.6f}")


if __name__ == "__main__":
    main()
