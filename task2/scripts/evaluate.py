import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate strategy1 model on test split and visualize training results.")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/runs/detect/runs/visdrone/yolov8m_1280_highscore/weights/best.pt",
        help="Path to trained model weights (best.pt).",
    )
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml", help="Dataset yaml path.")
    parser.add_argument(
        "--results_csv",
        type=str,
        default="runs/runs/detect/runs/visdrone/yolov8m_1280_highscore/results.csv",
        help="Training results.csv path.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0", help='Use "0" for GPU or "cpu".')
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="evaluation/strategy1")
    parser.add_argument("--run_name", type=str, default="test_eval")
    parser.add_argument("--save_json", action="store_true")
    return parser.parse_args()


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_training_curves(results_csv: Path, out_dir: Path):
    df = pd.read_csv(results_csv)

    # Robust epoch axis: if "epoch" starts from 1 in csv, keep it; otherwise fallback.
    if "epoch" in df.columns:
        x = df["epoch"]
    else:
        x = pd.Series(range(1, len(df) + 1))

    # Figure 1: Precision/Recall/mAP
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            ax.plot(x, df[col], label=col)
    ax.set_title("Strategy1 Detection Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_pr_map.png", dpi=200)
    plt.close(fig)

    # Figure 2: Train losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        if col in df.columns:
            ax.plot(x, df[col], label=col)
    ax.set_title("Strategy1 Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_loss.png", dpi=200)
    plt.close(fig)

    # Figure 3: Val losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        if col in df.columns:
            ax.plot(x, df[col], label=col)
    ax.set_title("Strategy1 Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss.png", dpi=200)
    plt.close(fig)

    # Figure 4: Learning rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ["lr/pg0", "lr/pg1", "lr/pg2"]:
        if col in df.columns:
            ax.plot(x, df[col], label=col)
    ax.set_title("Strategy1 Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate.png", dpi=200)
    plt.close(fig)

    # Save simple summary text
    best_map_idx = df["metrics/mAP50-95(B)"].idxmax() if "metrics/mAP50-95(B)" in df.columns else None
    summary_lines = []
    if best_map_idx is not None:
        summary_lines.append(f"Best mAP50-95 epoch: {int(df.loc[best_map_idx, 'epoch'])}")
        summary_lines.append(f"Best mAP50-95: {df.loc[best_map_idx, 'metrics/mAP50-95(B)']:.6f}")
        if "metrics/mAP50(B)" in df.columns:
            summary_lines.append(f"mAP50 at best epoch: {df.loc[best_map_idx, 'metrics/mAP50(B)']:.6f}")
        if "metrics/precision(B)" in df.columns:
            summary_lines.append(f"Precision at best epoch: {df.loc[best_map_idx, 'metrics/precision(B)']:.6f}")
        if "metrics/recall(B)" in df.columns:
            summary_lines.append(f"Recall at best epoch: {df.loc[best_map_idx, 'metrics/recall(B)']:.6f}")
    (out_dir / "training_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def run_test_eval(args, out_dir: Path):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from ultralytics import YOLO  # Imported here so CSV plotting can still run without ultralytics.

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(out_dir),
        name=args.run_name,
        save_json=args.save_json,
        plots=True,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)
    eval_summary = [
        f"save_dir: {save_dir}",
        f"mAP50: {metrics.box.map50:.6f}",
        f"mAP50-95: {metrics.box.map:.6f}",
        f"Precision: {metrics.box.mp:.6f}",
        f"Recall: {metrics.box.mr:.6f}",
    ]
    (out_dir / "test_eval_summary.txt").write_text("\n".join(eval_summary), encoding="utf-8")

    # Copy key eval artifacts to top-level out_dir for convenience.
    for fname in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.csv",
    ]:
        src = save_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # 1) Plot training csv results
    plot_training_curves(Path(args.results_csv), out_dir)

    # 2) Run test evaluation and produce confusion matrix
    run_test_eval(args, out_dir)

    print(f"Done. All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
