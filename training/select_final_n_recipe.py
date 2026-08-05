"""Select the stronger N1 recipe and expose deterministic fields to the refit shell script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--regularized", type=Path, required=True)
    parser.add_argument(
        "--field", required=True,
        choices=("recipe", "updates", "best_epoch", "selection_score"),
    )
    args = parser.parse_args()
    rows = {"control": load(args.control), "regularized": load(args.regularized)}
    recipe, report = max(
        rows.items(), key=lambda item: (float(item[1]["best_selection_score"]), item[0])
    )
    values = {
        "recipe": recipe,
        "best_epoch": int(report["best_epoch"]),
        "updates": int(report["best_epoch"]) * int(report["updates_per_epoch"]),
        "selection_score": float(report["best_selection_score"]),
    }
    print(values[args.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

