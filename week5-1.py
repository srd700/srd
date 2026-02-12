"""Compatibility entrypoint for coursework execution.

Some environments invoke this filename directly. This wrapper delegates to
`analysis_pipeline.run_pipeline` and preserves CLI options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the coursework pipeline (compatibility wrapper)."
    )
    parser.add_argument("--dataset1", default="studentdata1.xlsx", help="Path to studentdata1.xlsx")
    parser.add_argument("--dataset2", default="studentdata2.xlsx", help="Path to studentdata2.xlsx")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots for headless or dependency-limited environments.",
    )
    args = parser.parse_args()

    run_pipeline(Path(args.dataset1), Path(args.dataset2), enable_plots=not args.no_plots)


if __name__ == "__main__":
    main()
