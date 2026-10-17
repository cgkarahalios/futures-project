"""Execute the project's research notebooks in their dependency order.

Use --dry-run to inspect steps without downloading billable market data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
STAGES = (
    "01_collect_data.ipynb",
    "02_clean_features.ipynb",
    "03_validate_data.ipynb",
    "04_analyze_liquidity_migration.ipynb",
    "05_define_crossover.ipynb",
    "06_develop_early_indicators.ipynb",
    "07_predict_crossover.ipynb",
)


def selected_notebooks(start_at: int = 1, stop_at: int = 7) -> list[Path]:
    if not (1 <= start_at <= stop_at <= len(STAGES)):
        raise ValueError("Require 1 <= start_at <= stop_at <= 7")
    return [NOTEBOOKS / name for name in STAGES[start_at - 1:stop_at]]


def execute_notebook(path: Path, timeout: int = 1800) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    ).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--stop-at", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        paths = selected_notebooks(args.start_at, args.stop_at)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    for path in paths:
        if not path.is_file():
            parser.error(f"Notebook not found: {path}")
        print(f"{path.name}: {'planned' if args.dry_run else 'running'}", flush=True)
    if args.dry_run:
        return
    if args.start_at == 1 and not os.environ.get("DATABENTO_API_KEY"):
        parser.error("Set DATABENTO_API_KEY before downloading data (stage 1)")
    for path in paths:
        execute_notebook(path, timeout=args.timeout)
        print(f"{path.name}: complete", flush=True)


if __name__ == "__main__":
    main()
