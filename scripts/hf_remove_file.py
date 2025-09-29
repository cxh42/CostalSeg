import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


def remove_remote_file(repo_id: str, path_in_repo: str, repo_type: str = "dataset") -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.delete_file(
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Remove {path_in_repo}",
    )
    print(f"[remote] Deleted {repo_id}:{path_in_repo}")


def remove_local_file(local_root: str, path_in_repo: str) -> None:
    p = Path(local_root) / path_in_repo
    try:
        if p.exists():
            p.unlink()
            print(f"[local] Deleted {p}")
        else:
            print(f"[local] Not found: {p}")
    except Exception as e:
        print(f"[local] Failed to delete {p}: {e}")


def main(argv: list[str]) -> int:
    # Defaults for the known missing-mask sample
    repo_id = os.environ.get("HF_REPO", "AveMujica/CostalSeg-SJ")
    path = os.environ.get(
        "HF_PATH",
        "train/2024-09-07_13-21-00_SJ_jpg.rf.d2e929ae1f2aa77b490aef53871573ff.jpg",
    )
    repo_type = os.environ.get("HF_REPO_TYPE", "dataset")

    # Remote deletion (requires HF_TOKEN)
    try:
        remove_remote_file(repo_id, path, repo_type)
    except Exception as e:
        print(f"[remote] Warning: failed to delete remote file: {e}")
        print("Ensure HF_TOKEN is set or run `huggingface-cli login` then retry.")

    # Local mirrors (common locations used in this repo)
    remove_local_file("SegmentModelTraining/SilhouetteJaenette/dataset", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

