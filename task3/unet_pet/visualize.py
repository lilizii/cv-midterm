import csv
import os
from typing import Dict

import matplotlib.pyplot as plt

from .engine import RunResult


def save_plots(results: Dict[str, RunResult], out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    for mode, result in results.items():
        xs = [int(r['epoch']) for r in result.history]
        ys = [r['val_miou'] for r in result.history]
        plt.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5, label=mode.upper())
    plt.title('Validation mIoU Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'miou_comparison.png'), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for mode, result in results.items():
        xs = [int(r['epoch']) for r in result.history]
        ys = [r['train_loss'] for r in result.history]
        plt.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5, label=mode.upper())
    plt.title('Train Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_loss_comparison.png'), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for mode, result in results.items():
        xs = [int(r['epoch']) for r in result.history]
        ys = [r['val_loss'] for r in result.history]
        plt.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5, label=mode.upper())
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'val_loss_comparison.png'), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    modes = list(results.keys())
    ys = [results[mode].final_test_miou for mode in modes]
    colors = ['#4C78A8', '#F58518', '#54A24B'][: len(modes)]
    plt.bar([m.upper() for m in modes], ys, color=colors)
    plt.title('Final Test mIoU by Loss')
    plt.ylabel('mIoU')
    plt.ylim(0.0, min(1.0, max(ys) + 0.1 if ys else 1.0))
    plt.grid(axis='y', alpha=0.25)
    for idx, val in enumerate(ys):
        plt.text(idx, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'final_test_miou_bar.png'), dpi=180)
    plt.close()

    for mode, result in results.items():
        xs = [int(r['epoch']) for r in result.history]
        ys_miou = [r['val_miou'] for r in result.history]
        ys_loss = [r['train_loss'] for r in result.history]
        ys_val_loss = [r['val_loss'] for r in result.history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(xs, ys_miou, color='#0068B5', marker='o', linewidth=1.8, markersize=3.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Val mIoU', color='#0068B5')
        ax1.tick_params(axis='y', labelcolor='#0068B5')
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(xs, ys_loss, color='#D94F04', linestyle='--', linewidth=1.6)
        ax2.set_ylabel('Train Loss', color='#D94F04')
        ax2.tick_params(axis='y', labelcolor='#D94F04')
        plt.title(f'{mode.upper()} Training Curve')
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f'curve_{mode}.png'), dpi=180)
        plt.close(fig)

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys_loss, color='#D94F04', marker='o', linewidth=1.8, markersize=3.5)
        plt.title(f'{mode.upper()} Train Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'loss_{mode}.png'), dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys_loss, color='#D94F04', marker='o', linewidth=1.8, markersize=3.5, label='Train Loss')
        plt.plot(xs, ys_val_loss, color='#0068B5', marker='s', linewidth=1.8, markersize=3.5, label='Val Loss')
        plt.title(f'{mode.upper()} Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'loss_train_val_{mode}.png'), dpi=180)
        plt.close()


def save_summary(results: Dict[str, RunResult], out_dir: str) -> None:
    summary_path = os.path.join(out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['mode', 'best_val_miou', 'best_epoch', 'final_test_loss', 'final_test_miou'],
        )
        writer.writeheader()
        for mode, result in results.items():
            writer.writerow(
                {
                    'mode': mode,
                    'best_val_miou': f'{result.best_miou:.6f}',
                    'best_epoch': result.best_epoch,
                    'final_test_loss': f'{result.final_test_loss:.6f}',
                    'final_test_miou': f'{result.final_test_miou:.6f}',
                }
            )
