import argparse
import os

import pandas as pd

from unet_pet.engine import RunResult
from unet_pet.visualize import save_plots


def _load_history(csv_path: str):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                'epoch': float(r['epoch']),
                'train_loss': float(r['train_loss']),
                'val_loss': float(r['val_loss']),
                'val_miou': float(r['val_miou']),
                'lr': float(r['lr']),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description='Regenerate plots from metrics csv files.')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory with metrics_*.csv and summary.csv')
    parser.add_argument('--modes', nargs='+', default=['ce', 'dice', 'combo'], choices=['ce', 'dice', 'combo'])
    args = parser.parse_args()

    summary_path = os.path.join(args.out_dir, 'summary.csv')
    summary = pd.read_csv(summary_path)

    results = {}
    for mode in args.modes:
        metrics_path = os.path.join(args.out_dir, f'metrics_{mode}.csv')
        history = _load_history(metrics_path)
        row = summary[summary['mode'] == mode].iloc[0]
        results[mode] = RunResult(
            mode=mode,
            best_miou=float(row['best_val_miou']),
            best_epoch=int(row['best_epoch']),
            history=history,
            final_test_loss=float(row['final_test_loss']),
            final_test_miou=float(row['final_test_miou']),
        )

    save_plots(results, args.out_dir)
    print(f'Plots regenerated under: {args.out_dir}')


if __name__ == '__main__':
    main()
