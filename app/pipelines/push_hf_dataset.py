import argparse
import os

from app.config import settings
from app.core.hf_dataset import upload_dataset_assets


def main():
    parser = argparse.ArgumentParser(description="Upload Chroma + DailyMed name index to a HF dataset repo.")
    parser.add_argument("--repo-id", default="gylrt/medical-anchor-dataset")
    parser.add_argument("--token", default=settings.hf_token or os.getenv("HF_TOKEN", ""))
    parser.add_argument("--chroma-dir", default=settings.chroma_dir)
    parser.add_argument("--name-index", default=settings.dailymed_name_index_path)
    parser.add_argument("--public", action="store_true", help="Create repo as public if it does not exist.")
    args = parser.parse_args()

    upload_dataset_assets(
        repo_id=args.repo_id,
        token=args.token,
        chroma_dir=args.chroma_dir,
        name_index_path=args.name_index,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
