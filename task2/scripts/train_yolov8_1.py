import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone with high-score presets")
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="yolov8n/s/m/l/x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch", type=int, default=10, help="A10 30GB recommended start: 10 for yolov8m@1280")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="strategy1")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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
        optimizer="AdamW",
        lr0=0.003,
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
        mixup=0.05,
        copy_paste=0.0,
        amp=True,
        cache="ram",
        patience=30,
        save_period=10,
        val=True,
        plots=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
