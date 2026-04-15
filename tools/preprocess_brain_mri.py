import argparse
import random
from pathlib import Path

import cv2


def canonical_label(name: str):
    n = name.lower().strip()
    if n in {"yes", "tumor", "tumour", "positive"}:
        return "yes", 1
    if n in {"no", "notumor", "no_tumor", "negative"}:
        return "no", 0
    return None


def collect_labeled_images(root: Path):
    samples = []

    # Direct class folders under root.
    direct_candidates = [root / "yes", root / "Yes", root / "YES", root / "no", root / "No", root / "NO"]
    for d in direct_candidates:
        if d.is_dir():
            label_info = canonical_label(d.name)
            if not label_info:
                continue
            canon_name, label = label_info
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}:
                    samples.append((p, canon_name, label))

    # One-level child folders with label-like names.
    for child in root.iterdir():
        if not child.is_dir():
            continue
        label_info = canonical_label(child.name)
        if not label_info:
            continue
        canon_name, label = label_info
        for p in child.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}:
                samples.append((p, canon_name, label))

    # De-duplicate by resolved source path.
    dedup = {}
    for src, canon_name, label in samples:
        dedup[str(src.resolve())] = (src, canon_name, label)

    return list(dedup.values())


def to_pgm(src: Path, dst: Path, size: int):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), resized)
    return bool(ok)


def write_manifest(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# label path_to_pgm\n")
        for label, img_path in entries:
            f.write(f"{label} {img_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Brain MRI images with OpenCV into PGM files for C++ inference/training.")
    parser.add_argument("--input", default="data/brain_mri", help="Input dataset root containing yes/no folders")
    parser.add_argument("--output", default="data/brain_mri_pgm", help="Output folder for converted PGM files and manifests")
    parser.add_argument("--size", type=int, default=64, help="Square output size in pixels")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for split")
    parser.add_argument("--image", default="", help="Optional single image to preprocess for inference")
    parser.add_argument("--image-out", default="data/brain_mri_pgm/single_image.pgm", help="Output path for --image conversion")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise SystemExit(f"Input dataset path does not exist: {inp}")

    samples = collect_labeled_images(inp)
    if not samples:
        raise SystemExit("No readable image files found under input dataset path")

    converted = []
    for idx, (src, canon_name, label) in enumerate(samples):
        dst_name = f"{idx:06d}_{src.stem}.pgm"
        dst = out / canon_name / dst_name
        if to_pgm(src, dst, args.size):
            converted.append((label, str(dst.resolve())))

    if not converted:
        raise SystemExit("Image conversion failed for all files")

    rng = random.Random(args.seed)
    rng.shuffle(converted)

    train_count = int(len(converted) * args.train_ratio)
    train_count = max(1, min(train_count, len(converted) - 1)) if len(converted) > 1 else len(converted)

    train_entries = converted[:train_count]
    test_entries = converted[train_count:] if len(converted) > 1 else converted

    write_manifest(out / "train_manifest.txt", train_entries)
    write_manifest(out / "test_manifest.txt", test_entries)

    print(f"Converted samples: {len(converted)}")
    print(f"Train samples: {len(train_entries)}")
    print(f"Test samples: {len(test_entries)}")
    print(f"Train manifest: {(out / 'train_manifest.txt').resolve()}")
    print(f"Test manifest: {(out / 'test_manifest.txt').resolve()}")

    if args.image:
        single_src = Path(args.image)
        single_dst = Path(args.image_out)
        if not single_src.exists():
            raise SystemExit(f"--image path does not exist: {single_src}")
        if not to_pgm(single_src, single_dst, args.size):
            raise SystemExit(f"Failed to preprocess --image: {single_src}")
        print(f"Single image output: {single_dst.resolve()}")


if __name__ == "__main__":
    main()
