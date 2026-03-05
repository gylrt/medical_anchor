import os
from pathlib import Path


def upload_dataset_assets(
    repo_id: str,
    token: str,
    chroma_dir: str,
    name_index_path: str,
    private: bool = True,
):
    if not token:
        raise RuntimeError("Missing token. Set HF_TOKEN or pass --token.")

    chroma_path = Path(chroma_dir)
    index_path = Path(name_index_path)
    if not chroma_path.exists():
        raise RuntimeError(f"Missing chroma directory: {chroma_path}")
    if not index_path.exists():
        raise RuntimeError(f"Missing name index file: {index_path}")

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: poetry add huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    print(f"Uploading Chroma folder: {chroma_path} -> {repo_id}/chroma")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(chroma_path),
        path_in_repo="chroma",
        commit_message="Update Chroma dataset",
    )

    print(f"Uploading name index: {index_path} -> {repo_id}/processed/dailymed_name_index.json")
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(index_path),
        path_in_repo="processed/dailymed_name_index.json",
        commit_message="Update DailyMed name index",
    )

    print("Done.")


def _local_assets_present(chroma_dir: str, name_index_path: str) -> bool:
    chroma_path = Path(chroma_dir)
    index_path = Path(name_index_path)
    return (chroma_path / "chroma.sqlite3").exists() and index_path.exists()


def ensure_local_dataset_assets(
    repo_id: str,
    token: str,
    chroma_dir: str,
    name_index_path: str,
):
    if _local_assets_present(chroma_dir, name_index_path):
        print("Using local dataset assets.")
        return

    if not repo_id:
        raise RuntimeError(
            "Local dataset assets not found and no HF dataset repo configured."
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for remote fallback. Install with: poetry add \"huggingface-hub<1.0\""
        ) from exc

    chroma_path = Path(chroma_dir)
    index_path = Path(name_index_path)
    root = Path(os.path.commonpath([str(chroma_path.parent), str(index_path.parent)]))
    root.mkdir(parents=True, exist_ok=True)

    print(f"Local data missing. Downloading from HF dataset: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token or None,
        local_dir=str(root),
        allow_patterns=["chroma/**", "processed/dailymed_name_index.json"],
    )

    if not _local_assets_present(chroma_dir, name_index_path):
        raise RuntimeError(
            "HF dataset download completed, but required files are still missing: "
            f"{chroma_path / 'chroma.sqlite3'} and/or {index_path}"
        )

    print("Remote dataset assets downloaded and ready.")
