import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on VisDrone with head-only finetuning (freeze backbone/neck)."
    )
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="yolov8n/s/m/l/x.pt or checkpoint path")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="strategy3")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--freeze",
        type=int,
        default=22,
        help="Number of modules to freeze from model start. 22 usually keeps only detect head trainable for YOLOv8m.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="False",
        choices=["False", "ram", "disk"],
        help="Dataset cache mode. Use False/disk when RAM is limited.",
    )
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def parse_cache(v: str):
    if v == "False":
        return False
    return v


def main():
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mixup=0.03,
        copy_paste=0.0,
        amp=True,
        cache=parse_cache(args.cache),
        patience=args.patience,
        save_period=10,
        val=True,
        plots=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
