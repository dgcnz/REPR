from pathlib import Path
import os
import yaml


def find_run_id(wandb_dir: Path) -> str:
    """Return the wandb run ID extracted from the wandb directory."""
    if not wandb_dir.exists() or not wandb_dir.is_dir():
        raise FileNotFoundError(
            f"Expected to find a 'wandb' folder at {wandb_dir}, but it does not exist."
        )

    run_dirs = [p for p in wandb_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not run_dirs:
        raise RuntimeError(
            f"No subdirectory matching 'run-<timestamp>-<run_id>' found under {wandb_dir}."
        )

    latest_symlink = wandb_dir / "latest-run"
    if latest_symlink.is_symlink():
        target = Path(os.readlink(str(latest_symlink)))
        if target.exists() and target.name.startswith("run-"):
            run_dirs = [target]
        else:
            run_dirs = [run_dirs[0]]
    else:
        run_dirs = [run_dirs[0]]

    run_folder = run_dirs[0].name
    parts = run_folder.split("-", 2)
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected run-folder name: {run_folder}.")
    return parts[2]


def read_wandb_project_from_config(config_path: Path) -> str:
    """Load Hydra config.yaml and return logger.wandb.project."""
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        return cfg["logger"]["wandb"]["project"]
    except KeyError as exc:
        raise KeyError(
            f"Could not extract logger.wandb.project from {config_path}"
        ) from exc

