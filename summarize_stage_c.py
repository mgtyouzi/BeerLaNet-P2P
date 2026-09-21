import csv
import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_stage_c.py STAGE_C_ROOT")
    root = os.path.abspath(sys.argv[1])
    rows = []
    for mode in ("frozen", "joint"):
        experiment = os.path.join(root, f"beerlanet_{mode}")
        for domain, directory in (
            ("FFPE_source_test", "eval_ffpe_source_selection"),
            ("Frozen_diagnostic", "eval_frozen_diagnostic"),
        ):
            path = os.path.join(experiment, directory, "summary.json")
            with open(path, encoding="utf-8") as handle:
                summary = json.load(handle)
            rows.append(
                {
                    "experiment": mode,
                    "domain": domain,
                    "checkpoint_epoch": summary["checkpoint_epoch"],
                    "precision": summary["precision"],
                    "recall": summary["recall"],
                    "f1": summary["f1"],
                    "tp": summary["tp"],
                    "fp": summary["fp"],
                    "fn": summary["fn"],
                    "selection_role": summary["checkpoint_selection_role"],
                    "checkpoint": summary["checkpoint"],
                }
            )

    csv_path = os.path.join(root, "stage_c_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(root, "stage_c_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
