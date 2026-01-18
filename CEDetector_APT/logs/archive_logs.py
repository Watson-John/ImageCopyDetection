#!/usr/bin/env python3
"""Archive completed Slurm job logs into an archive directory."""
from __future__ import annotations

import argparse
import getpass
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def get_active_job_ids(user: str) -> Set[str]:
    """Return active job IDs for the given user using squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%A"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("squeue not found; assuming no active jobs.", file=sys.stderr)
        return set()
    except subprocess.CalledProcessError as exc:
        print(f"Failed to query squeue: {exc}", file=sys.stderr)
        return set()

    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def extract_job_id(path: Path) -> str | None:
    """Extract the trailing numeric job id from filenames like job_123 or foo_123.out."""
    match = re.search(r"_(\d+)(?:\.[^.]+)?$", path.name)
    return match.group(1) if match else None


def next_available(target: Path) -> Path:
    """Return a non-colliding path inside the archive directory."""
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    for idx in range(1, 1000):
        candidate = target.parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free name for {target}")


def plan_moves(logs_dir: Path, active_ids: Set[str]) -> List[Path]:
    """List log files/dirs that should be moved to archive."""
    to_move: List[Path] = []
    for entry in logs_dir.iterdir():
        if entry.name == "archive":
            continue
        job_id = extract_job_id(entry)
        if not job_id:
            continue
        if job_id in active_ids:
            continue
        to_move.append(entry)
    return to_move


def archive_logs(logs_dir: Path, dry_run: bool) -> Tuple[List[Path], Set[str]]:
    active_ids = get_active_job_ids(getpass.getuser())
    archive_dir = logs_dir / "archive"
    candidates = plan_moves(logs_dir, active_ids)

    moved: List[Path] = []
    if dry_run:
        return candidates, active_ids

    archive_dir.mkdir(exist_ok=True)
    for path in candidates:
        target = next_available(archive_dir / path.name)
        shutil.move(str(path), str(target))
        moved.append(target)

    return moved, active_ids


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive completed job logs into archive/.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the logs directory (defaults to this script's directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without performing the move",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    logs_dir: Path = args.logs_dir.resolve()

    if not logs_dir.exists():
        print(f"Logs dir not found: {logs_dir}", file=sys.stderr)
        return 1

    moved, active_ids = archive_logs(logs_dir, args.dry_run)

    print(f"Active job ids: {sorted(active_ids) if active_ids else 'none'}")
    if args.dry_run:
        print("Dry run; would move:")
        for path in moved:
            print(f"  {path}")
    else:
        if moved:
            print("Archived:")
            for path in moved:
                print(f"  {path}")
        else:
            print("Nothing to archive.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
