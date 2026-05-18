import argparse
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run tracker comparison (ByteTrack vs BoT-SORT)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--line", type=str, default="200,500,1000,500")
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


def run_once(tracker, model, source, line, max_frames):
    out_video = f"outputs/{tracker.replace('.yaml', '')}_tracking.mp4"
    out_evt = f"outputs/{tracker.replace('.yaml', '')}_events.csv"
    out_log = f"outputs/{tracker.replace('.yaml', '')}_track_log.csv"

    cmd = [
        "python", "scripts/track_and_count.py",
        "--model", model,
        "--source", source,
        "--output", out_video,
        "--tracker", tracker,
        "--line", line,
        "--save_csv", out_evt,
        "--save_track_csv", out_log,
    ]
    if max_frames > 0:
        cmd += ["--max_frames", str(max_frames)]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    text = p.stdout
    count = re.search(r"Crossed total:\s*(\d+)", text)
    fps = re.search(r"avg fps:\s*([0-9.]+)", text)
    return {
        "tracker": tracker,
        "crossed_total": int(count.group(1)) if count else -1,
        "avg_fps": float(fps.group(1)) if fps else -1.0,
        "video": out_video,
        "events": out_evt,
        "track_log": out_log,
    }


def main():
    args = parse_args()
    results = []
    for trk in ["bytetrack.yaml", "botsort.yaml"]:
        results.append(run_once(trk, args.model, args.source, args.line, args.max_frames))

    out = Path("outputs/tracker_compare_summary.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                f"tracker={item['tracker']} crossed_total={item['crossed_total']} avg_fps={item['avg_fps']} "
                f"video={item['video']} events={item['events']} track_log={item['track_log']}\n"
            )

    print("Tracker comparison finished:")
    for item in results:
        print(item)
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
