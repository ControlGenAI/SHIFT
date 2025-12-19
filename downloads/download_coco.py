import os
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm.auto import tqdm

def download_and_extract_zip(url: str, extract_to_dir: Path):
    """Downloads a zip file from a URL, extracts it, and cleans up the zip file."""
    extract_to_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download the file
    zip_path = extract_to_dir / url.split('/')[-1]
    print(f"Downloading {url} to {zip_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # 2. Extract the file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
            
        # 3. Clean up the zip file
        os.remove(zip_path)
        
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return

def download_coco_captions(data_dir: Path):
    """
    Downloads and organizes the COCO 2017 Caption annotations.
    
    Args:
        data_dir: The parent directory where the 'coco_captions_2017' folder will be created.
    """
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    target_dir = data_dir / "coco_captions_2017"
    
    download_and_extract_zip(url, target_dir)
    
    # The zip typically extracts to target_dir/annotations/ (e.g., target_dir/annotations/captions_train2017.json)
    annotations_dir = target_dir / "annotations"
    
    if annotations_dir.is_dir():
        print(f"Moving contents from {annotations_dir} to {target_dir}...")
        
        # Move all files/directories from the nested 'annotations' folder to the 'coco_captions_2017' folder
        for item in annotations_dir.iterdir():
            # Use shutil.move for safe cross-device moves
            shutil.move(str(item), str(target_dir / item.name)) 
        
        # Remove the now-empty nested 'annotations' folder
        print(f"Removing redundant folder {annotations_dir}...")
        shutil.rmtree(annotations_dir)
    
    print(f"COCO captions download and setup complete in: {target_dir}")

# ----------------------------------------------------------------------
# Example Usage:
# ----------------------------------------------------------------------

# Set your chosen directory
chosen_data_dir = Path("./data") 

if __name__ == '__main__':
    # Ensure the base directory exists
    chosen_data_dir.mkdir(parents=True, exist_ok=True)
    
    download_coco_captions(chosen_data_dir)