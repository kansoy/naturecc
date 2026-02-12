#!/usr/bin/env python3
"""Run the full replication pipeline."""

from src.figures import run_figures
from src.tables import run_tables


def main() -> None:
    run_tables()
    run_figures()
    print("pipeline complete")


if __name__ == "__main__":
    main()
