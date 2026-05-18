import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create a 2x2 panel for occlusion analysis")
    parser.add_argument("--frame_dir", type=str, default="outputs/occlusion_frames")
    parser.add_argument("--output", type=str, default="outputs/occlusion_panel.jpg")
    return parser.parse_args()


def add_title(img, text):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (img.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    args = parse_args()
    frame_dir = Path(args.frame_dir)
    imgs = sorted(frame_dir.glob("*.jpg"))[:4]
    if len(imgs) < 4:
        raise RuntimeError("Need at least 4 frames in frame_dir")

    frames = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Cannot read {p}")
        frames.append(add_title(im, p.stem))

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [cv2.resize(f, (w, h)) for f in frames]

    top = np.hstack([frames[0], frames[1]])
    bot = np.hstack([frames[2], frames[3]])
    panel = np.vstack([top, bot])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)
    print(f"Saved panel: {out}")


if __name__ == "__main__":
    main()
