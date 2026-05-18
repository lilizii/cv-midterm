import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Build a demo video from VisDrone test-dev images")
    parser.add_argument("--image_dir", type=str, default="archive/VisDrone2019-DET-test-dev/images")
    parser.add_argument("--output", type=str, default="outputs/demo_testdev_20s.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=20, help="10-30 seconds recommended")
    parser.add_argument("--prefix", type=str, default="", help="optional prefix filter, e.g. 0000074")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    imgs = sorted(image_dir.glob("*.jpg"))
    if args.prefix:
        imgs = [p for p in imgs if p.name.startswith(args.prefix)]

    if not imgs:
        raise RuntimeError("No images found for video generation")

    max_frames = args.fps * args.seconds
    imgs = imgs[:max_frames]

    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"Cannot read: {imgs[0]}")
    h, w = first.shape[:2]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    written = 0
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Output video: {output}")
    print(f"Frames: {written}, FPS: {args.fps}, Duration: {written/args.fps:.2f}s")


if __name__ == "__main__":
    main()
