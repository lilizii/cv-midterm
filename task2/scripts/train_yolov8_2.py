import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone with stronger presets")
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="yolov8n/s/m/l/x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=10, help="24GB GPU recommended: 8-10 for yolov8m@1280")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="strategy2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--freeze_epochs", type=int, default=8, help="Stage-1 frozen warmup epochs, 0 to disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_cls", action="store_true")
    parser.add_argument("--cache", type=str, default="False", help="False, ram, or disk")
    return parser.parse_args()


def parse_cache(cache_str: str):
    s = cache_str.strip().lower()
    if s in ("false", "none", "0", "off"):
        return False
    if s in ("ram", "disk"):
        return s
    raise ValueError("--cache must be one of: False, ram, disk")


def main():
    args = parse_args()
    cache_value = parse_cache(args.cache)
    model = YOLO(args.model)

    common = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0025,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0007,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=20,
        mosaic=1.0,
        mixup=0.08,
        copy_paste=0.0,
        degrees=3.0,
        translate=0.08,
        scale=0.5,
        shear=1.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.35,
        erasing=0.2,
        amp=True,
        cache=cache_value,
        multi_scale=True,
        val=True,
        plots=True,
        seed=args.seed,
        deterministic=False,
        save_period=10,
        patience=args.patience,
        single_cls=args.single_cls,
    )

    if args.resume:
        model.train(epochs=args.epochs, resume=True, **common)
        return

    if args.freeze_epochs > 0 and args.freeze_epochs < args.epochs:
        model.train(epochs=args.freeze_epochs, freeze=10, **common)

        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {last_ckpt}")

        model_stage2 = YOLO(str(last_ckpt))
        model_stage2.train(epochs=args.epochs, **common)
    else:
        model.train(epochs=args.epochs, **common)


if __name__ == "__main__":
    main()
