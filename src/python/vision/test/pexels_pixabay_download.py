import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# --- Pexels Download ---
def download_pexels(query="cat", per_page=10, output_dir="pexels_images"):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    os.makedirs(output_dir, exist_ok=True)
    for i, photo in enumerate(data.get("photos", []), 1):
        img_url = photo["src"]["original"]
        img_data = requests.get(img_url).content
        with open(f"{output_dir}/{query}_{i}.jpg", "wb") as f:
            f.write(img_data)
        print(f"[Pexels] Downloaded: {query}_{i}.jpg")

# --- Pixabay Download ---
def download_pixabay(query="cat", per_page=10, output_dir="pixabay_images"):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={per_page}"
    resp = requests.get(url)
    data = resp.json()
    os.makedirs(output_dir, exist_ok=True)
    for i, hit in enumerate(data.get("hits", []), 1):
        img_url = hit["largeImageURL"]
        img_data = requests.get(img_url).content
        with open(f"{output_dir}/{query}_{i}.jpg", "wb") as f:
            f.write(img_data)
        print(f"[Pixabay] Downloaded: {query}_{i}.jpg")
        time.sleep(1)  # ป้องกัน rate limit
    print("X-RateLimit-Remaining:", resp.headers.get("X-RateLimit-Remaining"))
    print("X-RateLimit-Reset:", resp.headers.get("X-RateLimit-Reset"))

if __name__ == "__main__":
    download_pexels(query="cat", per_page=10)
    download_pixabay(query="cat", per_page=10)
