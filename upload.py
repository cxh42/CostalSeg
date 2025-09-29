from huggingface_hub import HfApi, upload_folder
api = HfApi()
upload_folder(
    repo_id="AveMujica/CostalSeg-SJ",
    repo_type="dataset",
    folder_path="SegmentModelTraining/SilhouetteJaenette/dataset",
    path_in_repo="",
    allow_patterns=["train/*", "valid/*", "test/*", "*.txt"],
)
