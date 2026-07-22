#!/usr/bin/env python
"""List and delete run directories under outputs/.

Each run's directory name *is* its run_name (outputs/<run_name>/, see
docs/lifecycle.md "Run identity") -- there's no separate registry to keep in
sync, so this operates directly on the filesystem.

    python scripts/manage_runs.py list
    python scripts/manage_runs.py delete <run_name>
    python scripts/manage_runs.py delete <run_name> --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

OUTPUTS_ROOT = Path("outputs")


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def list_runs() -> None:
    if not OUTPUTS_ROOT.exists():
        print("No outputs/ directory yet -- no runs recorded.")
        return

    runs = sorted(
        (p for p in OUTPUTS_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print("No runs under outputs/.")
        return

    print(f"{'run_name':35s} {'modified':20s} {'size':>10s}")
    print("-" * 68)
    for run in runs:
        mtime = datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{run.name:35s} {mtime:20s} {_human_size(_dir_size(run)):>10s}")


def delete_run(run_name: str, skip_confirm: bool) -> None:
    run_dir = OUTPUTS_ROOT / run_name
    if not run_dir.exists():
        print(f"No such run: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if not skip_confirm:
        reply = input(f"Delete {run_dir} ({_human_size(_dir_size(run_dir))})? [y/N] ")
        if reply.strip().lower() != "y":
            print("Aborted.")
            return

    shutil.rmtree(run_dir)
    print(f"Deleted {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all run directories under outputs/")

    delete_parser = subparsers.add_parser("delete", help="Delete a run directory")
    delete_parser.add_argument("run_name", help="Name of the run to delete (its outputs/<run_name>/ directory)")
    delete_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    args = parser.parse_args()

    if args.command == "list":
        list_runs()
    elif args.command == "delete":
        delete_run(args.run_name, args.yes)


if __name__ == "__main__":
    main()
