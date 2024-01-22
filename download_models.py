import os
import urllib.request
import zipfile
from multiprocessing import Pool

def download_file(url_destination):
    url, destination = url_destination
    try:
        print(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {destination}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def rename_folder(src, dst):
    os.rename(src, dst)

def download_models():
    # Create directories if they don't exist
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./gfpgan/weights', exist_ok=True)

    # Download and unzip SadTalker
    sadtalker_zip_url = "https://github.com/OpenTalker/SadTalker/archive/refs/heads/main.zip"
    sadtalker_zip_destination = "SadTalker-main.zip"  # Save to current directory
    sadtalker_extract_path = "."

    download_file((sadtalker_zip_url, sadtalker_zip_destination))
    unzip_file(sadtalker_zip_destination, sadtalker_extract_path)
    rename_folder("SadTalker-main", "SadTalker")
    os.remove(sadtalker_zip_destination)

    # Define download URLs and destinations
    download_urls_destinations = [
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar", "./checkpoints/mapping_00109-model.pth.tar"),
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar", "./checkpoints/mapping_00229-model.pth.tar"),
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors", "./checkpoints/SadTalker_V0.0.2_256.safetensors"),
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors", "./checkpoints/SadTalker_V0.0.2_512.safetensors"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth", "./gfpgan/weights/alignment_WFLW_4HG.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", "./gfpgan/weights/detection_Resnet50_Final.pth"),
        ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "./gfpgan/weights/GFPGANv1.4.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth", "./gfpgan/weights/parsing_parsenet.pth")
    ]

    # Use multiprocessing to parallelize downloads
    with Pool() as pool:
        pool.map(download_file, download_urls_destinations)

    print("Downloads completed.")

if __name__ == "__main__":
    download_models()