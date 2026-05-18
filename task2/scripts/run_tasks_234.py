import os
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run task (2)(3)(4): tracking, line-counting, occlusion analysis.")
    parser.add_argument("--video", type=str, default="video.mp4")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/strategy1/weights/best.pt",
    )
    parser.add_argument("--line", type=str, default="200,500,1000,500")
    parser.add_argument("--tracker", type=str, default="configs/bytetrack_visdrone.yaml")
    parser.add_argument("--out_dir", type=str, default="detect")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--classes", type=str, default="")
    return parser.parse_args()


def run_cmd(cmd):
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracked_video = out_dir / "tracking_count.mp4"
    event_csv = out_dir / "crossing_events.csv"
    track_csv = out_dir / "track_log.csv"
    occ_dir = out_dir / "occlusion_analysis"

    cmd_track = [
        "python",
        "scripts1/track_and_count.py",
        "--model",
        args.model,
        "--source",
        args.video,
        "--output",
        str(tracked_video),
        "--tracker",
        args.tracker,
        "--line",
        args.line,
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--save_csv",
        str(event_csv),
        "--save_track_csv",
        str(track_csv),
    ]
    if args.classes.strip():
        cmd_track += ["--classes", args.classes]

    run_cmd(cmd_track)

    cmd_occ = [
        "python",
        "scripts1/analyze_occlusion_id.py",
        "--track_csv",
        str(track_csv),
        "--video",
        str(tracked_video),
        "--out_dir",
        str(occ_dir),
        "--window",
        "4",
    ]
    run_cmd(cmd_occ)

    print("All done.")
    print(f"Tracked video: {tracked_video}")
    print(f"Line crossing events: {event_csv}")
    print(f"Track log: {track_csv}")
    print(f"Occlusion panel/report: {occ_dir}")


if __name__ == "__main__":
    main()
