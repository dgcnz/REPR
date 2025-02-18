import wandb
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import LightningModule, LightningDataModule
from pathlib import Path

ARTIFACT_DIR = rootutils.find_root() / "artifacts"

def parse_artifact_name(full_name: str):
    # dgcnz/PARTv2/model-ukjrb3lq:v0
    entity, project, name = full_name.split("/")
    return entity, project, name

def download_artifact(full_name: str):
    api = wandb.Api()
    artifact = api.artifact(full_name)
    _, _, name = parse_artifact_name(full_name)
    out = artifact.download(root=ARTIFACT_DIR / name)
    return artifact, Path(out)

def download_config_file(entity: str, project: str, run_id: str) -> DictConfig:
    """Download a config file from wandb and return the path.
    :param entity: The entity name.
    :param project: The project name.
    :param run_id: The run id.

    :return: The config file as a DictConfig. Keys: ["model", "data", "trainer", "callbacks", "tags", etc.]
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return OmegaConf.create(run.config)

def get_datamodule_from_cfg(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    return hydra.utils.instantiate(cfg.data)


def get_model_and_data_modules_from_config(
    wandb_config: DictConfig,
) -> tuple[LightningModule, LightningDataModule]:
    """
    Instantiates the model and datamodule from a config file downloaded from wandb.
    :param wandb_config: The config file downloaded by `download_config_file`.
    """
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    model = hydra.utils.instantiate(wandb_config.model)
    datamodule = hydra.utils.instantiate(wandb_config.data)
    return model, datamodule


def get_cfg(overrides: list[str] = [], config_name: str = "train.yaml") -> DictConfig:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
        return cfg

