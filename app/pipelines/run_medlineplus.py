import argparse
from pathlib import Path

from app.sources.medlineplus import download as medline_download
from app.sources.medlineplus import ingest as medline_ingest


def run(mode: str, xml_path: str = ""):
    resolved_xml = Path(xml_path) if xml_path else None

    if mode in ("download", "all"):
        url = medline_download.find_latest_xml()
        resolved_xml = medline_download.download_file(url)
        medline_download.write_manifest(resolved_xml, url)

    if mode in ("ingest", "all"):
        if resolved_xml is None:
            candidates = sorted(Path("data/raw").glob("mplus_topics_*.xml"))
            if not candidates:
                raise RuntimeError("No MedlinePlus XML found in data/raw.")
            resolved_xml = candidates[-1]
        medline_ingest.ingest(str(resolved_xml))


def main():
    parser = argparse.ArgumentParser(description="Run MedlinePlus pipeline.")
    parser.add_argument(
        "--mode",
        choices=["download", "ingest", "all"],
        default="all",
    )
    parser.add_argument(
        "--xml-path",
        default="",
        help="Optional MedlinePlus XML path for ingest mode.",
    )
    args = parser.parse_args()
    run(mode=args.mode, xml_path=args.xml_path)


if __name__ == "__main__":
    main()
