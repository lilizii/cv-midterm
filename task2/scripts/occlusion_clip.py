import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract consecutive frames for occlusion analysis")
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument("--start_sec", type=float, required=True, help="start time in seconds")
    parser.add_argument("--num_frames", type=int, default=4, help="number of consecutive frames")
    parser.add_argument("--step", type=int, default=1, help="frame interval")
    parser.add_argument("--out_dir", type=str, default="outputs/occlusion_frames", help="output folder")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(args.start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = start_frame
    while saved < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        filename = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        saved += 1

        for _ in range(args.step - 1):
            ok2, _ = cap.read()
            frame_idx += 1
            if not ok2:
                break
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
