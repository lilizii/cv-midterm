import argparse
from pathlib import Path

import cv2

# VisDrone DET columns:
# bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to YOLO format")
    parser.add_argument("--root", type=str, default="archive", help="dataset root")
    parser.add_argument("--include_test", action="store_true", help="also convert test-dev annotations")
    return parser.parse_args()


def convert_split(split_root: Path):
    img_dir = split_root / "images"
    ann_dir = split_root / "annotations"
    label_dir = split_root / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        raise RuntimeError(f"No annotations found in: {ann_dir}")

    converted = 0
    for ann_path in ann_files:
        img_path = img_dir / (ann_path.stem + ".jpg")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        out_lines = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                x, y, bw, bh = map(float, parts[:4])
                score = int(float(parts[4]))
                category = int(float(parts[5]))

                # category 0 is ignored region in VisDrone DET.
                # keep only valid object categories 1..10 and score==1.
                if score != 1 or category < 1 or category > 10:
                    continue

                xc = (x + bw / 2.0) / w
                yc = (y + bh / 2.0) / h
                wn = bw / w
                hn = bh / h

                # clip to [0,1]
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                wn = min(max(wn, 0.0), 1.0)
                hn = min(max(hn, 0.0), 1.0)

                cls = category - 1
                out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_path = label_dir / ann_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        converted += 1

    print(f"[{split_root.name}] converted files: {converted}")


def main():
    args = parse_args()
    root = Path(args.root)

    train_root = root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    val_root = root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"

    convert_split(train_root)
    convert_split(val_root)

    if args.include_test:
        test_root = root / "VisDrone2019-DET-test-dev"
        convert_split(test_root)

    print("Done.")


if __name__ == "__main__":
    main()
