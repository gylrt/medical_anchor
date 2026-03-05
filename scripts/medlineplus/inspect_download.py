"""
Inspect script for download_medlineplus.py

Verifies that the downloaded XML and manifest are present and coherent.
Run after download_medlineplus.py to confirm the data layer is healthy.
"""

import json
from pathlib import Path

DATA_DIR = Path("data/raw")


def main():
    # --- Check manifest exists ---
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: manifest.json not found. Run download_medlineplus.py first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print("=== Manifest ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")

    # --- Check XML file exists ---
    xml_path = DATA_DIR / manifest["file"]
    if not xml_path.exists():
        print(f"\nERROR: XML file not found at {xml_path}")
        return

    print(f"\n=== XML File ===")
    print(f"  Path:        {xml_path}")
    print(f"  Size:        {xml_path.stat().st_size / 1_000_000:.2f} MB")

    # --- Verify checksum ---
    import hashlib
    h = hashlib.sha256()
    with open(xml_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual_hash = h.hexdigest()

    if actual_hash == manifest["sha256"]:
        print(f"  Checksum:    OK")
    else:
        print(f"  Checksum:    MISMATCH")
        print(f"    expected:  {manifest['sha256']}")
        print(f"    actual:    {actual_hash}")

    # --- Quick peek at raw XML ---
    print(f"\n=== XML Preview (first 500 chars) ===")
    with open(xml_path, encoding="utf-8") as f:
        print(f.read(500))


if __name__ == "__main__":
    main()
