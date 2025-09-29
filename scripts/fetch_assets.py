import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download


MODELS = {
    "MM_best_model.pth": "https://huggingface.co/AveMujica/CostalSeg-MM/resolve/main/MM_best_model.pth",
    "SJ_best_model.pth": "https://huggingface.co/AveMujica/CostalSeg-SJ/resolve/main/SJ_best_model.pth",
}

DATASETS = {
    # local_dir relative to repo root
    "AveMujica/CostalSeg-MM": "SegmentModelTraining/MetalMarcy/dataset",
    "AveMujica/CostalSeg-SJ": "SegmentModelTraining/SilhouetteJaenette/dataset",
}

REF_VECTORS = {
    # If missing locally, attempt to fetch from the HF Space repo
    "MM_mean.npy": "https://huggingface.co/spaces/AveMujica/CostalSegment/resolve/main/models/MM_mean.npy",
    "SJ_mean.npy": "https://huggingface.co/spaces/AveMujica/CostalSegment/resolve/main/models/SJ_mean.npy",
}


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def http_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)


def fetch_models(force: bool = False):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    for fname, url in MODELS.items():
        out = models_dir / fname
        if out.exists() and not force:
            print(f"[models] Exists: {out}")
            continue
        print(f"[models] Downloading {fname} ...")
        http_download(url, out)
        print(f"[models] Saved: {out} ({out.stat().st_size} bytes)")

    # Optional reference vectors
    for fname, url in REF_VECTORS.items():
        out = models_dir / fname
        if out.exists() and not force:
            print(f"[models] Exists: {out}")
            continue
        try:
            print(f"[models] Downloading {fname} ...")
            http_download(url, out)
            print(f"[models] Saved: {out} ({out.stat().st_size} bytes)")
        except Exception as e:
            # Non-fatal if vectors are unavailable
            print(f"[models] Warning: failed to download {fname}: {e}")


def fetch_datasets(force: bool = False):
    for repo_id, local_dir in DATASETS.items():
        print(f"[dataset] Sync {repo_id} -> {local_dir}")
        # If force, remove existing local_dir
        ld = Path(local_dir)
        if force and ld.exists():
            print(f"[dataset] Removing existing directory: {ld}")
            shutil.rmtree(ld)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        # Post-check
        for split in ("train", "valid", "test"):
            p = Path(local_dir) / split
            print(f"[dataset] {repo_id}:{split} -> {p} {'OK' if p.exists() else 'MISSING'}")

        # Prune any images that do not have a matching *_mask.png file
        prune_unpaired_images(local_dir)


def prune_unpaired_images(local_dir: str | Path) -> None:
    local_dir = Path(local_dir)
    removed = 0
    for split in ("train", "valid", "test"):
        sdir = local_dir / split
        if not sdir.exists():
            continue
        # Collect basenames
        img_paths = list(sdir.glob("*.jpg")) + list(sdir.glob("*.jpeg")) + list(sdir.glob("*.png"))
        # Exclude mask files from images list
        img_paths = [p for p in img_paths if not p.name.endswith("_mask.png")]
        mask_bases = {p.stem[:-5] if p.stem.endswith("_mask") else p.stem for p in sdir.glob("*_mask.png")}
        for ip in img_paths:
            base = ip.stem
            if base not in mask_bases:
                try:
                    ip.unlink()
                    removed += 1
                    print(f"[dataset] Removed unpaired image: {ip}")
                except Exception as e:
                    print(f"[dataset] Warning: failed to remove {ip}: {e}")
    if removed:
        print(f"[dataset] Pruned {removed} unpaired images without masks")


def main(argv: list[str]) -> int:
    force = "--force" in argv
    fetch_models(force=force)
    fetch_datasets(force=force)
    print("All assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
