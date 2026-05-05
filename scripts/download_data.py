#!/usr/bin/env python3
"""
Script to download ExamHandOCR dataset from Zenodo.
"""

import os
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


ZENODO_RECORD = "19145349"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

FILES = [
    "Current.zip",
    "Current_jst.zip",
    "annotations.zip",
    "metadata.zip",
]


def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        chunk_size: Download chunk size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded {output_path}")


def extract_zip(zip_path: str, extract_dir: str):
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to
    """
    import zipfile
    
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extracted to {extract_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Directory to download data to')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='Specific files to download (default: all)')
    parser.add_argument('--extract', action='store_true',
                       help='Extract zip files after download')
    parser.add_argument('--keep_zip', action='store_true',
                       help='Keep zip files after extraction')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = args.files if args.files else FILES
    
    # Download files
    for filename in files_to_download:
        url = f"{BASE_URL}/{filename}"
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"{filename} already exists, skipping download")
        else:
            print(f"Downloading {filename}...")
            try:
                download_file(url, str(output_path))
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue
        
        # Extract if requested
        if args.extract and filename.endswith('.zip'):
            extract_dir = output_dir / filename.replace('.zip', '')
            try:
                extract_zip(str(output_path), str(extract_dir))
                if not args.keep_zip:
                    output_path.unlink()
                    print(f"Removed {filename}")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
    
    print("\nDownload completed!")
    print(f"Data location: {output_dir}")


if __name__ == '__main__':
    main()
