import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze occlusion/crowded segment from track log and export 4-frame panel.")
    parser.add_argument("--track_csv", type=str, required=True, help="track log csv from track_and_count.py")
    parser.add_argument("--video", type=str, required=True, help="tracked output video path")
    parser.add_argument("--out_dir", type=str, default="outputs/occlusion_analysis")
    parser.add_argument("--window", type=int, default=4, help="number of consecutive frames")
    parser.add_argument("--distance_thr", type=float, default=80.0, help="center distance threshold for crowdedness")
    return parser.parse_args()


def load_track_rows(track_csv):
    frame_rows = defaultdict(list)
    with open(track_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fr = int(r["frame"])
            row = {
                "track_id": int(r["track_id"]),
                "class_name": r["class_name"],
                "cx": float(r["cx"]),
                "cy": float(r["cy"]),
                "conf": float(r["conf"]),
            }
            frame_rows[fr].append(row)
    return frame_rows


def frame_score(rows, distance_thr):
    n = len(rows)
    if n <= 1:
        return n, 0
    crowded_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = rows[i]["cx"] - rows[j]["cx"]
            dy = rows[i]["cy"] - rows[j]["cy"]
            if (dx * dx + dy * dy) ** 0.5 < distance_thr:
                crowded_pairs += 1
    return n, crowded_pairs


def pick_best_window(frame_rows, window, distance_thr):
    frames = sorted(frame_rows.keys())
    if not frames:
        raise RuntimeError("No rows found in track csv.")
    min_f, max_f = frames[0], frames[-1]
    best = None
    for start in range(min_f, max_f - window + 2):
        det_sum = 0
        crowd_sum = 0
        for f in range(start, start + window):
            n, c = frame_score(frame_rows.get(f, []), distance_thr)
            det_sum += n
            crowd_sum += c
        score = crowd_sum * 1000 + det_sum
        if best is None or score > best[0]:
            best = (score, start, det_sum, crowd_sum)
    return best[1], best[2], best[3]


def extract_frames(video_path, start_frame, window, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved = []
    for i in range(window):
        ok, frame = cap.read()
        if not ok:
            break
        fn = out_dir / f"frame_{start_frame + i:06d}.jpg"
        cv2.imwrite(str(fn), frame)
        saved.append(fn)
    cap.release()
    return saved


def make_panel(img_paths, out_path):
    ims = [cv2.imread(str(p)) for p in img_paths]
    ims = [im for im in ims if im is not None]
    if len(ims) < 4:
        return
    h = min(im.shape[0] for im in ims[:4])
    w = min(im.shape[1] for im in ims[:4])
    ims = [cv2.resize(im, (w, h)) for im in ims[:4]]
    panel = np.vstack([np.hstack([ims[0], ims[1]]), np.hstack([ims[2], ims[3]])])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def write_report(path, start_frame, window, det_sum, crowd_sum):
    text = []
    text.append("Occlusion/ID Analysis (Auto)")
    text.append(f"Selected frame window: [{start_frame}, {start_frame + window - 1}]")
    text.append(f"Total detections in window: {det_sum}")
    text.append(f"Crowded pairs (distance-based): {crowd_sum}")
    text.append("")
    text.append("Interpretation:")
    text.append("1) This window has high target density and frequent close interactions, so occlusion risk is high.")
    text.append("2) Check IDs in these 4 frames: if the same object keeps the same ID, tracking is stable.")
    text.append("3) If one object disappears and returns with a new ID, that is ID switch caused by occlusion/re-association.")
    path.write_text("\n".join(text), encoding="utf-8")


def main():
    args = parse_args()
    frame_rows = load_track_rows(args.track_csv)
    start_frame, det_sum, crowd_sum = pick_best_window(frame_rows, args.window, args.distance_thr)

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    panel_path = out_dir / "occlusion_panel.jpg"
    report_path = out_dir / "occlusion_report.txt"

    saved = extract_frames(args.video, start_frame, args.window, frames_dir)
    make_panel(saved, panel_path)
    write_report(report_path, start_frame, args.window, det_sum, crowd_sum)

    print(f"Saved frames: {frames_dir}")
    print(f"Saved panel: {panel_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
