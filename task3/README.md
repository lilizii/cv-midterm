# U-Net Pet Segmentation (Modular)

本项目将原始 `train_unet_pet.py` 拆分为模块化结构，分为数据处理、训练、测试和可视化四部分，便于维护和复现实验。

## 目录结构

```text
.
├── train.py                  # 训练入口
├── test.py                   # 测试入口
├── plot_metrics.py           # 从 csv 重画可视化图
├── train_unet_pet.py         # 兼容入口（内部调用 train.py）
├── unet_pet/
│   ├── __init__.py
│   ├── data.py               # 数据集、预处理、划分、随机种子
│   ├── model.py              # U-Net 网络结构
│   ├── losses.py             # CE / Dice / Combo 损失
│   ├── metrics.py            # mIoU 指标
│   ├── engine.py             # 训练与评估逻辑
│   └── visualize.py          # 训练过程图与 summary 导出
└── outputs*/                 # 训练输出示例
```

## 环境配置

1. 创建并激活环境（示例使用 conda）：

```bash
conda create -n unet-pet python=3.10 -y
conda activate unet-pet
```

2. 安装依赖：

```bash
pip install torch torchvision numpy matplotlib pandas tqdm
```

说明：
- 首次训练会自动下载 Oxford-IIIT Pet 到 `--data_root` 目录。
- 有 NVIDIA GPU 建议使用 CUDA 版 PyTorch，并在训练命令中加 `--amp`。

## 训练

### 1) 三种损失函数一起训练（CE / Dice / Combo）

```bash
python train.py \
  --data_root ./data \
  --out_dir ./outputs \
  --epochs 60 \
  --batch_size 8 \
  --image_size 256 \
  --lr 5e-4 \
  --base_ch 32 \
  --patience 10 \
  --amp \
  --modes ce dice combo
```

### 2) 指定组合损失权重（例如 CE:Dice = 1:2）

```bash
python train.py \
  --data_root ./data \
  --out_dir ./outputs_combo_1_3 \
  --epochs 60 \
  --batch_size 8 \
  --amp \
  --modes combo \
  --combo_ce_weight 1 \
  --combo_dice_weight 2
```

说明：
- `--combo_ce_weight` 与 `--combo_dice_weight` 只在 `mode=combo` 时生效。
- 兼容旧命令：`python train_unet_pet.py ...` 仍可用。

## 测试

使用某个已训练 checkpoint 在测试集评估：

```bash
python test.py \
  --ckpt ./outputs/unet_combo_best.pth \
  --data_root ./data \
  --image_size 256 \
  --batch_size 8 \
  --base_ch 32 \
  --mode combo \
  --combo_ce_weight 1 \
  --combo_dice_weight 2
```

输出示例：
- `[Test] loss=... mIoU=...`

## 可视化

训练结束会自动生成图像；也可从 CSV 重新生成：

```bash
python plot_metrics.py --out_dir ./outputs --modes ce dice combo
```

默认会输出：
- `miou_comparison.png`
- `train_loss_comparison.png`
- `val_loss_comparison.png`
- `final_test_miou_bar.png`
- `curve_<mode>.png`
- `loss_<mode>.png`
- `loss_train_val_<mode>.png`

## 输出文件说明

- `metrics_<mode>.csv`：每个 epoch 的 `train_loss/val_loss/val_miou/lr`
- `summary.csv`：每个 loss 的最佳验证集与最终测试集结果
- `unet_<mode>_best.pth`：最佳验证 mIoU 对应 checkpoint
