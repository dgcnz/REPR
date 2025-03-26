import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def plot_distribution(data: pd.DataFrame, label: str):
    sqrt_area = np.sqrt(data["height"] * data["width"])
    plt.figure(figsize=(8, 6))
    plt.hist(sqrt_area, bins=50, color="orchid", edgecolor="black")
    plt.title(f"{label} sqrt(Area) Distribution")
    plt.xlabel("sqrt(Area)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def print_stats(data: pd.DataFrame, label: str):
    sqrt_area = np.sqrt(data["height"] * data["width"])
    print(f"{label} Size Statistics:")
    print("  Height     -> min: {}, max: {}, median: {}".format(
        data["height"].min(), data["height"].max(), data["height"].median()
    ))
    print("  Width      -> min: {}, max: {}, median: {}".format(
        data["width"].min(), data["width"].max(), data["width"].median()
    ))
    print("  sqrt(Area) -> min: {}, max: {}, median: {}\n".format(
        sqrt_area.min(), sqrt_area.max(), sqrt_area.median()
    ))

def main():
    train_data = load_data("train_sizes.csv")
    val_data = load_data("val_sizes.csv")
    
    print_stats(train_data, "Train")
    print_stats(val_data, "Validation")
    
    plot_distribution(train_data, "Train")
    plot_distribution(val_data, "Validation")

if __name__ == "__main__":
    main()