import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 tracking + line crossing count + report logs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/tracking_count.mp4")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--line", type=str, default="200,500,1000,500")
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--save_csv", type=str, default="outputs/crossing_events.csv")
    parser.add_argument("--save_track_csv", type=str, default="outputs/track_log.csv")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means full video")
    return parser.parse_args()


def side_of_line(px, py, x1, y1, x2, y2):
    val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return np.sign(val)


def parse_line(line_str):
    vals = [int(v.strip()) for v in line_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--line must be x1,y1,x2,y2")
    return vals


def parse_classes(classes_str):
    if not classes_str:
        return None
    return [int(x.strip()) for x in classes_str.split(",") if x.strip()]


def main():
    args = parse_args()
    model = YOLO(args.model)

    x1, y1, x2, y2 = parse_line(args.line)
    classes = parse_classes(args.classes)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_last_side = {}
    crossed_ids = set()
    crossing_events = []
    track_log = []
    frame_idx = 0

    t0 = time.time()
    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            verbose=False,
        )

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for box, tid, cls_id, conf in zip(boxes, ids, clss, confs):
                    x_min, y_min, x_max, y_max = box
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2

                    cur_side = side_of_line(cx, cy, x1, y1, x2, y2)
                    prev_side = track_last_side.get(tid, cur_side)

                    crossed = 0
                    if prev_side != 0 and cur_side != 0 and prev_side != cur_side:
                        crossed = 1
                        if tid not in crossed_ids:
                            crossed_ids.add(tid)
                            crossing_events.append([frame_idx, tid, cls_id, float(conf), cx, cy])

                    track_last_side[tid] = cur_side

                    color = (0, 255, 0) if tid not in crossed_ids else (0, 165, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.circle(frame, (cx, cy), 3, color, -1)
                    cls_name = model.names.get(int(cls_id), str(cls_id)) if isinstance(model.names, dict) else str(cls_id)
                    label = f"ID:{tid} {cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x_min, max(20, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    track_log.append([
                        frame_idx, tid, cls_id, cls_name, float(conf), x_min, y_min, x_max, y_max, cx, cy, int(cur_side), crossed
                    ])

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, f"Tracker: {args.tracker}", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Crossed Count: {len(crossed_ids)}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        writer.write(frame)
        frame_idx += 1

    elapsed = time.time() - t0
    cap.release()
    writer.release()

    csv_path = Path(args.save_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "track_id", "class_id", "conf", "center_x", "center_y"])
        wcsv.writerows(crossing_events)

    track_csv_path = Path(args.save_track_csv)
    track_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(track_csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "track_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "line_side", "crossed_flag"])
        wcsv.writerows(track_log)

    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Done. Output video: {output_path}")
    print(f"Crossed total: {len(crossed_ids)}")
    print(f"Event csv: {csv_path}")
    print(f"Track log csv: {track_csv_path}")
    print(f"Processed frames: {frame_idx}, elapsed: {elapsed:.2f}s, avg fps: {proc_fps:.2f}")


if __name__ == "__main__":
    main()
