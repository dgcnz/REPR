import subprocess
# conda run --no-capture-output  -n hummingbird python -m scripts.eval model=partmae_v6
# conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
#   model=partmae_v6 \
#   data=voc-mini \
#   'model.pretrained_cfg_overlay.state_dict.state_dict.f="outputs/2025-05-28/11-34-14/epoch_0024.ckpt","outputs/2025-05-28/11-34-14/epoch_0049.ckpt","outputs/2025-05-28/11-34-14/epoch_0074.ckpt","outputs/2025-05-28/11-34-14/epoch_0099.ckpt"'
# HYDRA_FULL_ERROR=1 conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
# model=partmae_v6 \
# data=voc \
# model.pretrained_cfg_overlay.state_dict.state_dict.f=$(ls -d outputs/2025-05-29/12-10-52/*  | grep epoch | paste -sd, -)

def eval(cfg: dict):
    """
    Evaluate a model checkpoint using the eval script.
    grep the 'mIoU' score from the output of the eval script

    example output from eval script:
    ...
    mIoU: 0.2343
    
    :param ckpt_path : Path to the model checkpoint.
    """
    # Override with any additional configuration
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "hummingbird",
        "python", "-m", "scripts.run_hummingbird",
        *[f"{k}={v}" for k, v in cfg.items()]
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running eval script:", result.stderr)
        return None
    # Extract mIoU from the output
    for line in result.stdout.splitlines():
        if "mIoU:" in line:
            mIoU = float(line.split("mIoU:")[1].strip())
            return {
                "mIoU": mIoU
            }