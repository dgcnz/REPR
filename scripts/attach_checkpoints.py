#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import wandb
import yaml

def find_run_id(wandb_dir: Path) -> str:
    """
    Search for a folder named run-<timestamp>-<run_id> inside wandb_dir,
    and return the <run_id> part. Raises if not found or ambiguous.
    """
    if not wandb_dir.exists() or not wandb_dir.is_dir():
        raise FileNotFoundError(f"Expected to find a 'wandb' folder at {wandb_dir}, but it does not exist.")

    run_dirs = [p for p in wandb_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not run_dirs:
        raise RuntimeError(f"No subdirectory matching 'run-<timestamp>-<run_id>' found under {wandb_dir}.")

    # If multiple run-* dirs exist, prefer the one pointed to by "latest-run" symlink
    latest_symlink = wandb_dir / "latest-run"
    if latest_symlink.is_symlink():
        target = Path(os.readlink(str(latest_symlink)))
        if target.exists() and target.name.startswith("run-"):
            run_dirs = [target]
        else:
            run_dirs = [run_dirs[0]]
    else:
        run_dirs = [run_dirs[0]]

    run_folder = run_dirs[0].name  # e.g. "run-20250603_153622-xd04s7ey"
    parts = run_folder.split("-", 2)
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected run-folder name: {run_folder}.")
    return parts[2]

def read_wandb_project_from_config(config_path: Path) -> str:
    """
    Load the Hydra config.yaml and return logger.wandb.project.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        return cfg["logger"]["wandb"]["project"]
    except KeyError:
        raise KeyError(f"Could not extract logger.wandb.project from {config_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Resume a finished W&B run by ID and create one artifact per checkpoint file."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory of the previous run (e.g. 'outputs/2025-06-03/15-36-21')."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    wandb_dir   = output_dir / "wandb"
    config_path = output_dir / ".hydra" / "config.yaml"

    try:
        run_id  = find_run_id(wandb_dir)
    except Exception as e:
        print(f"ERROR: could not determine run_id: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        project = read_wandb_project_from_config(config_path)
    except Exception as e:
        print(f"ERROR: could not read project name from config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    # Resume the original run; rely on WANDB_ENTITY and WANDB_MODE env vars if needed.
    print(f"Resuming run ID='{run_id}' in project='{project}'")
    run = wandb.init(
        id=run_id,
        resume="allow",
        project=project
    )

    # Find all checkpoint files matching step_*.ckpt
    ckpt_files = sorted(output_dir.glob("step_*.ckpt"))
    if not ckpt_files:
        print(f"WARNING: no files matching 'step_*.ckpt' found under {output_dir}.", file=sys.stderr)

    # For each checkpoint, create and log a separate artifact
    for ckpt in ckpt_files:
        abs_path = ckpt.resolve()
        artifact_name = f"model-{ckpt.stem}"   # e.g. "model-step_0000500"
        artifact = wandb.Artifact(artifact_name, type="model")
        uri = f"file://{abs_path}"
        artifact.add_reference(uri, name=ckpt.name)
        print(f"Logging artifact '{artifact_name}' referencing '{ckpt.name}' → {uri}")
        run.log_artifact(artifact)

    # If there's a 'last.ckpt', create a separate artifact for it as well
    last_ckpt = output_dir / "last.ckpt"
    if last_ckpt.exists():
        abs_last = last_ckpt.resolve()
        artifact_name = "model-last"
        artifact = wandb.Artifact(artifact_name, type="model")
        uri = f"file://{abs_last}"
        artifact.add_reference(uri, name="last.ckpt")
        print(f"Logging artifact '{artifact_name}' referencing 'last.ckpt' → {uri}")
        run.log_artifact(artifact)

    print("Finished logging all checkpoint artifacts.")
    run.finish()
    print("Done.")

if __name__ == "__main__":
    main()
