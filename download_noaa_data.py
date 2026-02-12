"""
Download and decompress NOAA Storm Events data for 2023.

This script downloads the raw NOAA Storm Events details data directly from NCEI,
decompresses it, and saves it to our Bronze layer (data/01_raw/noaa/).
"""

import gzip
import os
import shutil
from pathlib import Path

import pandas as pd
import requests


def download_noaa_storm_events_2023() -> str:
    """
    Download 2023 NOAA Storm Events details data.
    
    Returns:
        str: Path to the downloaded and decompressed CSV file
    """
    # File details
    url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2023_c20260116.csv.gz"
    
    # Paths
    raw_data_dir = Path("data/01_raw/noaa")
    compressed_file = raw_data_dir / "StormEvents_details_2023.csv.gz"
    final_file = raw_data_dir / "StormEvents_details_2023.csv"
    
    # Ensure directory exists
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading NOAA Storm Events 2023 data...")
    print(f"Source: {url}")
    print(f"Target: {final_file}")
    
    # Download the compressed file
    print("Downloading compressed file...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(compressed_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {compressed_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Decompress the file
    print("Decompressing file...")
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(final_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the compressed file to save space
    compressed_file.unlink()
    
    print(f"Decompressed to {final_file}")
    print(f"Final file size: {final_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return str(final_file)


def explore_data(file_path: str) -> None:
    """
    Load and explore the downloaded NOAA data.
    
    Args:
        file_path: Path to the CSV file
    """
    print("\n" + "="*60)
    print("NOAA STORM EVENTS 2023 - DATA EXPLORATION")
    print("="*60)
    
    # Load data
    print("Loading data into pandas DataFrame...")
    df = pd.read_csv(file_path)
    
    print(f"\nBASIC INFO")
    print(f"Row count: {len(df):,}")
    print(f"Column count: {len(df.columns)}")
    
    print(f"\nCOLUMN NAMES")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2}. {col}")
    
    print(f"\nDATA TYPES")
    print(df.dtypes.to_string())
    
    print(f"\nFIRST 5 ROWS")
    print(df.head().to_string())
    
    print(f"\nNULL COUNTS")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(null_counts) > 0:
        print(null_counts.to_string())
    else:
        print("No null values found!")
    
    print(f"\nEVENT TYPES")
    if 'EVENT_TYPE' in df.columns:
        event_types = df['EVENT_TYPE'].value_counts()
        print(f"Unique event types: {len(event_types)}")
        print(event_types.to_string())
    else:
        print("EVENT_TYPE column not found")
    
    print(f"\nCZ TYPES (County/Zone Types)")
    if 'CZ_TYPE' in df.columns:
        cz_types = df['CZ_TYPE'].value_counts()
        print(f"Unique CZ types: {len(cz_types)}")
        print(cz_types.to_string())
    else:
        print("CZ_TYPE column not found")
    
    print(f"\nDAMAGE_PROPERTY EXAMPLES")
    if 'DAMAGE_PROPERTY' in df.columns:
        damage_samples = df['DAMAGE_PROPERTY'].dropna().head(20)
        print(f"Sample values (showing format we'll need to parse):")
        for i, val in enumerate(damage_samples, 1):
            print(f"{i:2}. '{val}'")
    else:
        print("DAMAGE_PROPERTY column not found")


if __name__ == "__main__":
    # Download and process the data
    csv_file = download_noaa_storm_events_2023()
    
    # Explore the data
    explore_data(csv_file)