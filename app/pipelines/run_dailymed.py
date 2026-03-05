import argparse
import sys

from app.sources.dailymed import download as dailymed_download
from app.sources.dailymed import ingest as dailymed_ingest


def _run_download(
    mode: str,
    output_jsonl: str,
    cleanup_zip: bool,
    month: str,
):
    argv = ["run_dailymed_download", "--mode", mode, "--output-jsonl", output_jsonl, "--month", month]
    if cleanup_zip:
        argv.append("--cleanup-zip")
    old_argv = sys.argv
    try:
        sys.argv = argv
        dailymed_download.main()
    finally:
        sys.argv = old_argv


def run(
    mode: str,
    input_jsonl: str,
    download_mode: str,
    dedup_threshold: float,
    cleanup_zip: bool,
    month: str,
):
    if mode in ("download", "all"):
        _run_download(
            mode=download_mode,
            output_jsonl=input_jsonl,
            cleanup_zip=cleanup_zip,
            month=month,
        )

    if mode in ("ingest", "all"):
        dailymed_ingest.ingest_dailymed(
            input_jsonl,
            dedup=True,
            dedup_similarity_threshold=dedup_threshold,
            require_boxed_warning_similarity=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Run DailyMed pipeline.")
    parser.add_argument(
        "--mode",
        choices=["download", "ingest", "all"],
        default="all",
    )
    parser.add_argument(
        "--download-mode",
        choices=["auto", "monthly", "full-all"],
        default="auto",
        help="Mode passed to DailyMed download step.",
    )
    parser.add_argument(
        "--input-jsonl",
        default="data/processed/dailymed_minimal.jsonl",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--cleanup-zip",
        action="store_true",
    )
    parser.add_argument(
        "--month",
        default="",
        help="Optional month token for monthly mode (e.g. feb2026).",
    )
    args = parser.parse_args()
    run(
        mode=args.mode,
        input_jsonl=args.input_jsonl,
        download_mode=args.download_mode,
        dedup_threshold=args.dedup_threshold,
        cleanup_zip=args.cleanup_zip,
        month=args.month or dailymed_download.datetime.utcnow().strftime("%b%Y").lower(),
    )


if __name__ == "__main__":
    main()
