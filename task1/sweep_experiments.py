import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import List

import pandas as pd

PY = "python"


@dataclass
class Exp:
    name: str
    args: List[str]


def run(cmd: List[str]):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def read_summary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fmt_float(x: float) -> str:
    return f"{x:.0e}" if x < 1e-2 else f"{x:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--baseline_model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final_epochs", type=int, default=50)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    parser.add_argument("--weight_decays", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    args = parser.parse_args()

    ensure_dir(args.runs_root)
    rows = []

    # Stage-1: Baseline pretrained grid search
    print("\n=== Stage-1: Baseline grid search (pretrained) ===")
    baseline_grid = []
    for lr in args.lrs:
        for wd in args.weight_decays:
            name = f"grid_{args.baseline_model}_pretrained_lr_{fmt_float(lr)}_wd_{fmt_float(wd)}"
            exp = Exp(
                name=name,
                args=[
                    "--model_name", args.baseline_model,
                    "--pretrained",
                    "--epochs", str(args.epochs),
                    "--lr", str(lr),
                    "--weight_decay", str(wd),
                    "--backbone_lr_mult", str(args.backbone_lr_mult),
                    "--freeze_backbone_epochs", str(args.freeze_backbone_epochs),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    "--val_split", str(args.val_split),
                ],
            )
            baseline_grid.append((exp, lr, wd))

    best = {"val_acc": -1.0, "name": None, "lr": None, "weight_decay": None}
    for exp, lr, wd in baseline_grid:
        out_dir = os.path.join(args.runs_root, exp.name)
        cmd = [PY, "train.py", "--output_dir", out_dir] + exp.args
        run(cmd)
        summary = read_summary(os.path.join(out_dir, "summary.json"))
        val_acc = float(summary.get("best_val_acc", 0.0))
        test_acc = float(summary.get("final_test_acc", 0.0))
        rows.append([exp.name, "baseline_grid", val_acc, test_acc, lr, wd, True, args.baseline_model])
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "name": exp.name, "lr": lr, "weight_decay": wd}

    print(
        f"Best baseline: {best['name']} | val_acc={best['val_acc']:.4f} "
        f"| lr={best['lr']} | weight_decay={best['weight_decay']}"
    )

    # Stage-2: Compare models using best hyperparameters from baseline grid
    print("\n=== Stage-2: Model comparison with best baseline hyperparameters ===")
    lr_star = best["lr"]
    wd_star = best["weight_decay"]

    compare_exps = [
        Exp(
            name=f"baseline_best_{args.baseline_model}_pretrained",
            args=[
                "--model_name", args.baseline_model,
                "--pretrained",
                "--epochs", str(args.final_epochs),
                "--lr", str(lr_star),
                "--weight_decay", str(wd_star),
                "--backbone_lr_mult", str(args.backbone_lr_mult),
                "--freeze_backbone_epochs", str(args.freeze_backbone_epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--val_split", str(args.val_split),
            ],
        ),
        Exp(
            name=f"ablation_scratch_{args.baseline_model}",
            args=[
                "--model_name", args.baseline_model,
                "--epochs", str(args.final_epochs),
                "--lr", str(lr_star),
                "--weight_decay", str(wd_star),
                "--backbone_lr_mult", "1.0",
                "--freeze_backbone_epochs", "0",
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--val_split", str(args.val_split),
            ],
        ),
        Exp(
            name="attn_se_resnet18_pretrained",
            args=[
                "--model_name", "resnet18_se",
                "--pretrained",
                "--epochs", str(args.final_epochs),
                "--lr", str(lr_star),
                "--weight_decay", str(wd_star),
                "--backbone_lr_mult", str(args.backbone_lr_mult),
                "--freeze_backbone_epochs", str(args.freeze_backbone_epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--val_split", str(args.val_split),
            ],
        ),
        Exp(
            name="attn_cbam_resnet18_pretrained",
            args=[
                "--model_name", "resnet18_cbam",
                "--pretrained",
                "--epochs", str(args.final_epochs),
                "--lr", str(lr_star),
                "--weight_decay", str(wd_star),
                "--backbone_lr_mult", str(args.backbone_lr_mult),
                "--freeze_backbone_epochs", str(args.freeze_backbone_epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--val_split", str(args.val_split),
            ],
        ),
        Exp(
            name="lightweight_transformer_swin_t_pretrained",
            args=[
                "--model_name", "swin_t",
                "--pretrained",
                "--epochs", str(args.final_epochs),
                "--lr", str(lr_star),
                "--weight_decay", str(wd_star),
                "--backbone_lr_mult", str(args.backbone_lr_mult),
                "--freeze_backbone_epochs", "2",
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--val_split", str(args.val_split),
            ],
        ),
    ]

    for exp in compare_exps:
        out_dir = os.path.join(args.runs_root, exp.name)
        cmd = [PY, "train.py", "--output_dir", out_dir] + exp.args
        run(cmd)
        summary = read_summary(os.path.join(out_dir, "summary.json"))
        val_acc = float(summary.get("best_val_acc", 0.0))
        test_acc = float(summary.get("final_test_acc", 0.0))
        model_name = exp.args[exp.args.index("--model_name") + 1]
        pretrained = "--pretrained" in exp.args
        rows.append([exp.name, "model_compare", val_acc, test_acc, lr_star, wd_star, pretrained, model_name])

    df = pd.DataFrame(
        rows,
        columns=["exp_name", "stage", "best_val_acc", "final_test_acc", "lr", "weight_decay", "pretrained", "model_name"],
    )
    df = df.sort_values(by="best_val_acc", ascending=False)

    leaderboard_path = os.path.join(args.runs_root, "leaderboard.csv")
    df.to_csv(leaderboard_path, index=False)

    best_cfg_path = os.path.join(args.runs_root, "best_baseline_hparams.json")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_model": args.baseline_model,
                "best_baseline_exp": best["name"],
                "best_baseline_val_acc": best["val_acc"],
                "best_lr": lr_star,
                "best_weight_decay": wd_star,
                "grid_lrs": args.lrs,
                "grid_weight_decays": args.weight_decays,
                "search_epochs": args.epochs,
                "final_epochs": args.final_epochs,
                "val_split": args.val_split,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nAll experiments finished.")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Best baseline hyperparameters: {best_cfg_path}")
    print(df)


if __name__ == "__main__":
    main()
