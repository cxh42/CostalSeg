import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# Reuse model and helpers from the Gradio app
from app import (
    MODEL_PATHS,
    REFERENCE_IMAGE_DIRS,
    REFERENCE_VECTOR_PATHS,
    CLASSES,
    load_model,
    preprocess_image,
    generate_segmentation_map,
    create_overlay,
    load_reference_images,
    load_reference_vector,
)
from pipeline.ImgOutlier import detect_outliers
from pipeline.normalization import align_images


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_class_percentages(mask: np.ndarray) -> Dict[str, float]:
    total = mask.size
    out: Dict[str, float] = {}
    for idx, name in enumerate(CLASSES):
        cnt = int((mask == idx).sum())
        if cnt > 0:
            out[name] = round(cnt * 100.0 / total, 3)
    # Sort by percent desc for stable CSV ordering (optional)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def save_outputs(
    seg_map_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    out_base: str,
    save_seg: bool,
    save_overlay: bool,
):
    if save_seg:
        seg_bgr = cv2.cvtColor(seg_map_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_base + "_seg.png", seg_bgr)
    if save_overlay:
        ovl_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_base + "_overlay.png", ovl_bgr)


def write_csv_header(fh):
    # Dynamic header with classes; plus filename and outlier flag if present later
    cols = ["filename"] + CLASSES + ["outlier"]
    fh.write(",".join(cols) + "\n")


def write_csv_row(fh, filename: str, percentages: Dict[str, float], outlier: str):
    row = [filename]
    # Fill in class order deterministically (missing -> 0)
    for cls in CLASSES:
        row.append(str(percentages.get(cls, 0.0)))
    row.append(outlier)
    fh.write(",".join(row) + "\n")


def list_images(input_dir: str, recursive: bool = False) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files: List[str] = []
    if recursive:
        for ext in exts:
            files.extend(glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            files.extend(glob(os.path.join(input_dir, ext)))
    files.sort()
    return files


def infer_image(
    model, device: str, image_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run segmentation inference on a single image path.

    Returns (mask_indices, seg_map_rgb, overlay_rgb)
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(rgb)
    with torch.no_grad():
        pred = model(image_tensor.to(device))
    seg_map = generate_segmentation_map(pred, orig_h, orig_w)  # RGB
    overlay = create_overlay(rgb, seg_map)
    mask = pred.argmax(1).squeeze().detach().cpu().numpy().astype(np.uint8)
    return mask, seg_map, overlay


def cmd_segment(args: argparse.Namespace) -> int:
    location = args.location
    input_dir = args.input
    output_dir = args.output
    recursive = args.recursive
    save_seg = not args.no_seg
    save_overlay = args.overlay
    do_outlier = args.outlier

    if location not in MODEL_PATHS:
        print(f"Unknown location: {location}. Choose from: {list(MODEL_PATHS.keys())}")
        return 2

    model_path = MODEL_PATHS[location]
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Please place the weights as documented in README.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    if model is None:
        print("Failed to load model.")
        return 3

    # Prepare outlier reference if requested
    ref_vector = []
    ref_images: List[np.ndarray] = []
    if do_outlier:
        vec_path = REFERENCE_VECTOR_PATHS.get(location)
        if vec_path and os.path.exists(vec_path):
            ref_vector = load_reference_vector(vec_path)
        ref_dir = REFERENCE_IMAGE_DIRS.get(location)
        if ref_dir:
            ref_images = load_reference_images(ref_dir)
        if not ref_images:
            print("Warning: No reference images found for outlier detection; proceeding without references.")

    files = list_images(input_dir, recursive=recursive)
    if not files:
        print(f"No images found in: {input_dir}")
        return 0

    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        write_csv_header(fh)
        for idx, path in enumerate(files, 1):
            try:
                mask, seg_map, overlay = infer_image(model, device, path)
                perc = compute_class_percentages(mask)

                # Determine outlier status if enabled
                outlier_flag = "NA"
                if do_outlier and ref_images:
                    # detect_outliers expects lists of BGR images
                    bgr = cv2.imread(path)
                    if bgr is None:
                        raise RuntimeError(f"Failed to read image: {path}")
                    if isinstance(ref_vector, np.ndarray) and ref_vector.size > 0:
                        filtered, _ = detect_outliers(ref_images, [bgr], ref_vector)
                    else:
                        filtered, _ = detect_outliers(ref_images, [bgr])
                    is_outlier = len(filtered) == 0
                    outlier_flag = "1" if is_outlier else "0"

                # Save images
                base = os.path.splitext(os.path.basename(path))[0]
                out_base = os.path.join(output_dir, base)
                save_outputs(seg_map, overlay, out_base, save_seg, save_overlay)

                write_csv_row(fh, os.path.basename(path), perc, outlier_flag)
                if idx % 10 == 0 or idx == len(files):
                    print(f"Processed {idx}/{len(files)} images...")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"Done. Results written to: {output_dir}\nSummary CSV: {csv_path}")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    location = args.location
    reference = args.reference
    input_dir = args.input
    output_dir = args.output
    recursive = args.recursive
    save_seg = not args.no_seg
    save_overlay = args.overlay

    if location not in MODEL_PATHS:
        print(f"Unknown location: {location}. Choose from: {list(MODEL_PATHS.keys())}")
        return 2

    model_path = MODEL_PATHS[location]
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Please place the weights as documented in README.")
        return 2

    if not os.path.exists(reference):
        print(f"Reference image not found: {reference}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    if model is None:
        print("Failed to load model.")
        return 3

    ref_bgr = cv2.imread(reference)
    if ref_bgr is None:
        print(f"Failed to read reference image: {reference}")
        return 3

    files = list_images(input_dir, recursive=recursive)
    if not files:
        print(f"No images found in: {input_dir}")
        return 0

    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        write_csv_header(fh)
        for idx, path in enumerate(files, 1):
            try:
                tgt_bgr = cv2.imread(path)
                if tgt_bgr is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                aligned, _ = align_images([ref_bgr, tgt_bgr], [np.zeros_like(ref_bgr), np.zeros_like(tgt_bgr)])
                aligned_tgt_bgr = aligned[1]

                # Run inference on aligned image
                rgb = cv2.cvtColor(aligned_tgt_bgr, cv2.COLOR_BGR2RGB)
                image_tensor, orig_h, orig_w = preprocess_image(rgb)
                with torch.no_grad():
                    pred = model(image_tensor.to(device))
                seg_map = generate_segmentation_map(pred, orig_h, orig_w)
                overlay = create_overlay(rgb, seg_map)
                mask = pred.argmax(1).squeeze().detach().cpu().numpy().astype(np.uint8)
                perc = compute_class_percentages(mask)

                # Save outputs
                base = os.path.splitext(os.path.basename(path))[0]
                out_base = os.path.join(output_dir, base)
                save_outputs(seg_map, overlay, out_base, save_seg, save_overlay)

                write_csv_row(fh, os.path.basename(path), perc, outlier="NA")
                if idx % 10 == 0 or idx == len(files):
                    print(f"Processed {idx}/{len(files)} images...")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"Done. Results written to: {output_dir}\nSummary CSV: {csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch processing for CoastalSeg: segmentation and spatial alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # segment subcommand
    ps = sub.add_parser("segment", help="Batch segment images")
    ps.add_argument("--location", choices=list(MODEL_PATHS.keys()), required=True, help="Select model by site")
    ps.add_argument("--input", required=True, help="Input directory of images")
    ps.add_argument("--output", required=True, help="Output directory")
    ps.add_argument("--recursive", action="store_true", help="Recursively search for images")
    ps.add_argument("--overlay", action="store_true", help="Save overlay image in addition to segmentation map")
    ps.add_argument("--no-seg", action="store_true", help="Do not save segmentation map image")
    ps.add_argument("--outlier", action="store_true", help="Run outlier detection and include in CSV")
    ps.set_defaults(func=cmd_segment)

    # align subcommand
    pa = sub.add_parser("align", help="Batch spatial alignment + segmentation")
    pa.add_argument("--location", choices=list(MODEL_PATHS.keys()), required=True, help="Select model by site")
    pa.add_argument("--reference", required=True, help="Reference image path")
    pa.add_argument("--input", required=True, help="Input directory of target images to align + segment")
    pa.add_argument("--output", required=True, help="Output directory")
    pa.add_argument("--recursive", action="store_true", help="Recursively search for images")
    pa.add_argument("--overlay", action="store_true", help="Save overlay image in addition to segmentation map")
    pa.add_argument("--no-seg", action="store_true", help="Do not save segmentation map image")
    pa.set_defaults(func=cmd_align)

    return p


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
