import requests
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json

BASE_URL = "https://medlineplus.gov/xml/mplus_topics_{date}.xml"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_xml(max_days_back=30):
    """Try today, then go back up to max_days_back."""
    today = datetime.utcnow().date()

    for i in range(max_days_back):
        d = today - timedelta(days=i)
        date_str = d.isoformat()

        url = BASE_URL.format(date=date_str)
        print(f"Checking: {url}")

        r = requests.head(url, timeout=10)

        if r.status_code == 200:
            print(f"Found latest XML: {url}")
            return url

    raise RuntimeError("No XML file found in last days.")


def download_file(url):
    filename = url.split("/")[-1]
    output_path = DATA_DIR / filename

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(r.content)

    print("Downloaded:", output_path)
    return output_path


def write_manifest(path, url):
    manifest = {
        "source": "MedlinePlus Health Topics XML",
        "file": path.name,
        "url": url,
        "downloaded_at": datetime.utcnow().isoformat(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }

    manifest_path = DATA_DIR / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest updated.")


def main():
    url = find_latest_xml()
    file_path = download_file(url)
    write_manifest(file_path, url)


if __name__ == "__main__":
    main()