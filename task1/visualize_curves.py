import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def find_experiment_logs(runs_root: str) -> List[str]:
    log_paths = []
    for root, _, files in os.walk(runs_root):
        if "train_log.csv" in files:
            log_paths.append(os.path.join(root, "train_log.csv"))
    return sorted(log_paths)


def safe_read_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def plot_single_experiment(df: pd.DataFrame, exp_name: str, out_dir: str):
    required_cols = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[WARN] Skip {exp_name}: missing columns. Found {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=2)
    axes[0].set_title(f"{exp_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train_acc", linewidth=2)
    axes[1].plot(df["epoch"], df["val_acc"], label="val_acc", linewidth=2)
    axes[1].set_title(f"{exp_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{exp_name}_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_val_acc(experiment_frames, out_dir: str):
    if not experiment_frames:
        return

    plt.figure(figsize=(11, 6))
    for exp_name, df in experiment_frames:
        if {"epoch", "val_acc"}.issubset(set(df.columns)):
            plt.plot(df["epoch"], df["val_acc"], label=exp_name, linewidth=1.8)

    plt.title("Validation Accuracy Comparison (All Experiments)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_experiments_val_acc.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--output_dir", type=str, default="runs/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_paths = find_experiment_logs(args.runs_root)
    if not log_paths:
        print(f"No train_log.csv found under {args.runs_root}")
        return

    experiment_frames = []
    for log_path in log_paths:
        exp_name = os.path.basename(os.path.dirname(log_path))
        df = safe_read_csv(log_path)
        if df is None:
            continue

        plot_single_experiment(df, exp_name, args.output_dir)
        experiment_frames.append((exp_name, df))

    plot_all_val_acc(experiment_frames, args.output_dir)

    print(f"Done. Plots saved to: {args.output_dir}")
    print(f"Total experiments visualized: {len(experiment_frames)}")


if __name__ == "__main__":
    main()
