import os
import requests
from dotenv import load_dotenv
import time
import json

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

ANIMALS = [
    ("cow", "วัว"),
    ("pig", "หมู"),
    ("chicken", "ไก่"),
    ("duck", "เป็ด"),
    ("goat", "แพะ"),
    ("sheep", "แกะ"),
    ("buffalo", "ควาย")
]

PEXELS_PER_CLASS = 10
PIXABAY_PER_CLASS = 10

# --- Download from Pexels ---
def download_pexels(animal_en, output_dir):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={animal_en}&per_page={PEXELS_PER_CLASS}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, photo in enumerate(data.get("photos", []), 1):
        img_url = photo["src"]["original"]
        fname = f"{animal_en}_pexels_{i}.jpg"
        fpath = os.path.join(output_dir, fname)
        img_data = requests.get(img_url).content
        with open(fpath, "wb") as f:
            f.write(img_data)
        print(f"[Pexels] {animal_en}: {fname}")
        paths.append(fpath)
    return paths

# --- Download from Pixabay ---
def download_pixabay(animal_en, output_dir):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={animal_en}&image_type=photo&per_page={PIXABAY_PER_CLASS}"
    resp = requests.get(url)
    data = resp.json()
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, hit in enumerate(data.get("hits", []), 1):
        img_url = hit["largeImageURL"]
        fname = f"{animal_en}_pixabay_{i}.jpg"
        fpath = os.path.join(output_dir, fname)
        img_data = requests.get(img_url).content
        with open(fpath, "wb") as f:
            f.write(img_data)
        print(f"[Pixabay] {animal_en}: {fname}")
        paths.append(fpath)
        time.sleep(1)
    return paths

if __name__ == "__main__":
    dataset = []
    for animal_en, animal_th in ANIMALS:
        # Download from Pexels
        pexels_dir = os.path.join("pexels_images", animal_en)
        pexels_paths = download_pexels(animal_en, pexels_dir)
        # Download from Pixabay
        pixabay_dir = os.path.join("pixabay_images", animal_en)
        pixabay_paths = download_pixabay(animal_en, pixabay_dir)
        # Add to dataset
        for p in pexels_paths + pixabay_paths:
            dataset.append({
                "image_path": os.path.relpath(p, start=os.getcwd()),
                "label_en": animal_en,
                "label_th": animal_th
            })
    # Save as JSONL
    os.makedirs("data/output", exist_ok=True)
    out_path = f"data/output/vision_animals_{int(time.time())}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nDataset saved: {out_path}")
