"""Print the selected integer epoch from a train_v2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    epoch = report.get("best_epoch")
    if not epoch:
        history = report.get("history") or []
        if not history:
            raise SystemExit("training report không có history")
        epoch = max(history, key=lambda row: row["selection_score"])["epoch"]
    print(int(epoch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
