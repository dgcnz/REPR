#!/usr/bin/env python3
"""
cleanup_wandb.py

Traverse a directory tree, locate local WandB run folders by their date/time stamped directories
(e.g., outputs/YYYY-MM-DD/HH-MM-SS), detect the actual run subfolder under `wandb`, and delete the
entire time-stamped directory if the run ID is no longer present in the WandB project on the server.

- Entity defaults to 'dgcnz'.
- Each run's project is read from that run's .hydra/config.yaml under logger.wandb.project via OmegaConf.
- Lists all candidates for deletion first and prompts per-candidate confirmation.

Usage:
    python cleanup_wandb.py \
        --root /path/to/checkpoints \
        [--entity your_entity] \
        [--dry-run]

Requirements:
    pip install wandb omegaconf
"""
import shutil
import argparse
from pathlib import Path
from wandb import Api
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove local WandB run folders for runs deleted on the server"
    )
    p.add_argument(
        "--entity", default="dgcnz",
        help="WandB entity (user or team), default 'dgcnz'"
    )
    p.add_argument(
        "--root", required=True,
        help="Root directory to scan for date-stamped run folders"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List candidates without deleting anything"
    )
    return p.parse_args()


def fetch_live_run_ids(entity: str, project: str) -> set:
    api = Api()
    path = f"{entity}/{project}"
    print(f"Fetching live runs for {path}...")
    runs = api.runs(path)
    ids = {r.id for r in runs}
    print(f"Found {len(ids)} live runs in project '{project}'.")
    return ids


def find_project(time_dir: Path) -> str:
    cfg_path = time_dir / '.hydra' / 'config.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    try:
        project = cfg.logger.wandb.project
    except Exception:
        raise ValueError(f"Could not find 'logger.wandb.project' in {cfg_path}")
    if not isinstance(project, str):
        raise ValueError(f"Invalid project name in {cfg_path}: {project}")
    return project


def scan_and_cleanup(root: Path, entity: str, dry_run: bool):
    live_cache = {}
    candidates = []

    # Collect all candidates for removal
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir()):
            wandb_dir = time_dir / 'wandb'
            if not wandb_dir.is_dir():
                continue
            run_folder = next((e for e in wandb_dir.iterdir()
                               if e.is_dir() and (e.name.startswith('run-') or e.name.startswith('offline-run-'))), None)
            if not run_folder:
                print(f"Skipping {time_dir}: no run subfolder in wandb/")
                continue
            runid = run_folder.name.rsplit('-', 1)[-1]
            try:
                project = find_project(time_dir)
            except Exception as e:
                print(f"Skipping {time_dir}: {e}")
                continue
            if project not in live_cache:
                live_cache[project] = fetch_live_run_ids(entity, project)
            if runid not in live_cache[project]:
                candidates.append((time_dir, project, runid))

    if not candidates:
        print("No run directories to remove.")
        return

    print("The following run directories are candidates for removal:")
    for td, proj, rid in candidates:
        print(f"  {td}  (project={proj}, run_id={rid})")

    if dry_run:
        print("\nDry run mode: no directories will be removed.")
        return

    # Per-candidate confirmation and removal
    removed = []
    for td, proj, rid in candidates:
        resp = input(f"\nDelete {td}? [y/N]: ").strip().lower()
        if resp in ('y', 'yes'):
            try:
                print(f"Removing {td}...")
                shutil.rmtree(td)
                removed.append(td)
            except Exception as e:
                print(f"Error removing {td}: {e}")
        else:
            print(f"Skipped {td}.")

    print(f"\nTotal removed: {len(removed)} of {len(candidates)} candidates.")


if __name__ == '__main__':
    args = parse_args()
    scan_and_cleanup(Path(args.root), args.entity, args.dry_run)
