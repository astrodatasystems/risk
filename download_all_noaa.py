"""Download NOAA Storm Events details files for years 2010-2024.

Bronze layer: raw data exactly as received from NOAA, only decompressed.
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
OUTPUT_DIR = Path("data/01_raw/noaa")

# Exact filenames scraped from NOAA directory listing (creation dates vary by year)
FILES = {
    2010: "StormEvents_details-ftp_v1.0_d2010_c20250520.csv.gz",
    2011: "StormEvents_details-ftp_v1.0_d2011_c20250520.csv.gz",
    2012: "StormEvents_details-ftp_v1.0_d2012_c20250520.csv.gz",
    2013: "StormEvents_details-ftp_v1.0_d2013_c20250520.csv.gz",
    2014: "StormEvents_details-ftp_v1.0_d2014_c20250520.csv.gz",
    2015: "StormEvents_details-ftp_v1.0_d2015_c20251118.csv.gz",
    2016: "StormEvents_details-ftp_v1.0_d2016_c20250818.csv.gz",
    2017: "StormEvents_details-ftp_v1.0_d2017_c20260116.csv.gz",
    2018: "StormEvents_details-ftp_v1.0_d2018_c20260116.csv.gz",
    2019: "StormEvents_details-ftp_v1.0_d2019_c20260116.csv.gz",
    2020: "StormEvents_details-ftp_v1.0_d2020_c20260116.csv.gz",
    2021: "StormEvents_details-ftp_v1.0_d2021_c20250520.csv.gz",
    2022: "StormEvents_details-ftp_v1.0_d2022_c20250721.csv.gz",
    2023: "StormEvents_details-ftp_v1.0_d2023_c20260116.csv.gz",
    2024: "StormEvents_details-ftp_v1.0_d2024_c20260116.csv.gz",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for year, filename in sorted(FILES.items()):
    url = f"{BASE_URL}/{filename}"
    gz_path = OUTPUT_DIR / filename
    csv_path = OUTPUT_DIR / f"storm_events_details_{year}.csv"

    if csv_path.exists():
        print(f"  [{year}] Already exists, skipping: {csv_path.name}")
        continue

    print(f"  [{year}] Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, gz_path)
    except Exception as e:
        print(f"  [{year}] FAILED to download: {e}")
        continue

    # Decompress .gz -> .csv
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Remove the .gz file to save space
    gz_path.unlink()
    print(f"  [{year}] Saved: {csv_path.name}")

print("\nDone.")
