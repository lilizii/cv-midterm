# Oxford-IIIT Pet: 迁移学习、超参数搜索与注意力机制实验

## 1. 安装

```bash
pip install torch torchvision pandas
```

## 2. 数据划分与评估协议

当前代码使用标准 `train/val/test` 流程：
- 先从官方 `trainval` 划分出 `train` 与 `val`（默认 `val_split=0.2`）。
- 训练过程只在 `val` 上选最优模型（`best_val_acc`）。
- 训练结束后加载最优 checkpoint，仅在 `test` 上做一次最终评估（`final_test_acc`）。

这样可避免在超参数搜索阶段“用测试集选参”。

## 3. 单次训练（可选）

```bash
python train.py --model_name resnet18 --pretrained --epochs 30 --lr 1e-3 --weight_decay 1e-4 --backbone_lr_mult 0.1 --freeze_backbone_epochs 3 --val_split 0.2 --output_dir runs/baseline_manual
```

训练策略：
- 分类头（`fc` / `head`）从零开始训练。
- backbone 使用预训练参数并以更小学习率微调（`backbone_lr_mult`）。
- 可先冻结 backbone 若干 epoch，仅训练分类头。

## 4. 一键实验

```bash
python sweep_experiments.py
```

默认流程分两阶段：
- Stage-1：仅对 pretrained baseline 做网格搜索，自动找最优 `lr` 与 `weight_decay`（按 `best_val_acc` 选优）。
- Stage-2：使用 Stage-1 最优超参数训练并对比：
  - `baseline_best_*_pretrained`
  - `ablation_scratch_*`
  - `attn_se_resnet18_pretrained`
  - `attn_cbam_resnet18_pretrained`
  - `lightweight_transformer_swin_t_pretrained`

## 5. 可调参数

```bash
python sweep_experiments.py \
  --baseline_model resnet34 \
  --epochs 30 \
  --val_split 0.2 \
  --lrs 1e-4 3e-4 1e-3 3e-3 \
  --weight_decays 1e-5 1e-4 1e-3
```

## 6. 输出结果

- `runs/leaderboard.csv`：所有实验结果（默认按 `best_val_acc` 排序）。
- `runs/best_baseline_hparams.json`：Baseline 网格搜索最优参数记录。
- 每个实验目录下：
  - `summary.json`：`best_val_acc`、`best_val_epoch`、`final_test_acc`。
  - `train_log.csv`：逐 epoch 训练与验证指标（`train_*`、`val_*`）。
  - `best.pt`：按验证集最优保存的模型权重。

## 7. 代码结构

- `train.py`：统一训练入口（含 `train/val/test` 划分与评估）。
- `models_attention.py`：SE / CBAM 与注意力 ResNet 封装。
- `sweep_experiments.py`：两阶段实验调度脚本。
