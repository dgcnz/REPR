import pandas as pd
from datasets import load_dataset
import multiprocessing

def get_image_sizes(batch):
    return {
        "height": [img.height for img in batch["image"]],
        "width": [img.width for img in batch["image"]],
    }

def process_split(split_name: str, csv_name: str):
    ds = load_dataset("ILSVRC/imagenet-1k", split=split_name)
    ds = ds.map(get_image_sizes, batched=True, batch_size=64, num_proc=8, remove_columns=["image"])
    # Force asynchronous processing to complete
    df = pd.DataFrame({"height": ds["height"], "width": ds["width"]})
    df.to_csv(csv_name, index=False)
    print(f"Exported {csv_name}")

def main():
    process_split("train", "train_sizes.csv")
    process_split("validation", "val_sizes.csv")

if __name__ == "__main__":
    main()