import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import json
import argparse
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

def bing_image_search(query, num_images=5, outdir="scraped_images"):
    """
    ค้นหารูปภาพจาก Bing Images (scraping) โดยเติม 'free' ต่อท้าย query
    """
    search_query = f"{query} free"
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(search_query)}&form=HDRSC2&first=1&tsc=ImageBasicHover"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    os.makedirs(outdir, exist_ok=True)
    img_tags = soup.find_all("a", class_="iusc")
    count = 0
    for tag in img_tags:
        m = tag.get("m")
        if not m:
            continue
        try:
            m_json = json.loads(m)
            img_url = m_json.get("murl")
            if not img_url:
                continue
            img_data = requests.get(img_url, timeout=10).content
            fname = os.path.join(outdir, f"{query.replace(' ', '_')}_{count+1}.jpg")
            with open(fname, "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {fname}")
            count += 1
            if count >= num_images:
                break
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] {e}")
    if count == 0:
        print("No images found or download failed.")

def duckduckgo_image_search(query, num_images=5, outdir="scraped_images"):
    """
    ค้นหารูปภาพจาก DuckDuckGo Images (scraping)
    """
    import re
    search_query = f"{query} free"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    # Get vqd token
    resp = requests.get(f"https://duckduckgo.com/?q={urllib.parse.quote(search_query)}&iax=images&ia=images", headers=headers)
    m = re.search(r'vqd=([\d-]+)&', resp.text)
    if not m:
        m = re.search(r"vqd='([\d-]+)'", resp.text)
    if not m:
        print("[ERROR] Cannot find vqd token for DuckDuckGo!")
        return
    vqd = m.group(1)
    # Get image results (JSON)
    url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(search_query)}&vqd={vqd}&f=,,,&p=1"
    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        results = data.get("results", [])
    except Exception as e:
        print(f"[ERROR] DuckDuckGo image JSON: {e}")
        return
    os.makedirs(outdir, exist_ok=True)
    count = 0
    for item in results:
        img_url = item.get("image")
        if not img_url:
            continue
        try:
            img_data = requests.get(img_url, timeout=10).content
            fname = os.path.join(outdir, f"{query.replace(' ', '_')}_duck_{count+1}.jpg")
            with open(fname, "wb") as f:
                f.write(img_data)
            print(f"[DuckDuckGo] Downloaded: {fname}")
            count += 1
            if count >= num_images:
                break
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] {e}")
    if count == 0:
        print("No images found or download failed (DuckDuckGo).")

# --- Pexels ---
def pexels_image_urls(query, num_images=10):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page={num_images}"
    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        photos = data.get("photos", [])
        return [p["src"]["original"] for p in photos if "src" in p and "original" in p["src"]]
    except Exception as e:
        print(f"[ERROR] Pexels: {e}")
        return []

# --- Pixabay ---
def pixabay_image_urls(query, num_images=10):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={urllib.parse.quote(query)}&image_type=photo&per_page={num_images}"
    try:
        resp = requests.get(url)
        data = resp.json()
        hits = data.get("hits", [])
        return [h["largeImageURL"] for h in hits if "largeImageURL" in h]
    except Exception as e:
        print(f"[ERROR] Pixabay: {e}")
        return []

# --- Bing (url only) ---
def bing_image_urls(query, num_images=10):
    search_query = f"{query} free"
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(search_query)}&form=HDRSC2&first=1&tsc=ImageBasicHover"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        img_tags = soup.find_all("a", class_="iusc")
        urls = []
        for tag in img_tags:
            m = tag.get("m")
            if not m:
                continue
            try:
                m_json = json.loads(m)
                img_url = m_json.get("murl")
                if img_url:
                    urls.append(img_url)
                if len(urls) >= num_images:
                    break
            except Exception:
                continue
        return urls
    except Exception as e:
        print(f"[ERROR] Bing: {e}")
        return []

# --- DuckDuckGo (url only) ---
def duckduckgo_image_urls(query, num_images=10):
    import re
    import random
    search_query = f"{query} free"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"}
    session = requests.Session()
    try:
        resp = session.get(f"https://duckduckgo.com/?q={urllib.parse.quote(search_query)}&iax=images&ia=images", headers=headers, timeout=10)
        vqd = None
        patterns = [
            r'vqd=([\d-]+)&',
            r"vqd='([\d-]+)'",
            r'vqd=([\d]+)',
            r'window.vqd ?= ?"([\d]+)"',
        ]
        for pat in patterns:
            m = re.search(pat, resp.text)
            if m:
                vqd = m.group(1)
                break
        if not vqd:
            # fallback: หาใน script tag
            scripts = re.findall(r'<script.*?>(.*?)</script>', resp.text, re.DOTALL)
            for script in scripts:
                m = re.search(r'window.vqd ?= ?"([\d]+)"', script)
                if m:
                    vqd = m.group(1)
                    break
        if not vqd:
            print("[ERROR] Cannot find vqd token for DuckDuckGo! (HTML structure changed)")
            return []
        url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(search_query)}&vqd={vqd}&f=,,,&p=1"
        tries = 0
        results = []
        while tries < 3 and len(results) < num_images:
            try:
                resp = session.get(url, headers=headers, timeout=10, allow_redirects=True)
                data = resp.json()
                results.extend(data.get("results", []))
                if "next" in data and data["next"]:
                    url = "https://duckduckgo.com" + data["next"]
                else:
                    break
            except Exception as e:
                print(f"[ERROR] DuckDuckGo image JSON: {e}")
                break
            tries += 1
        img_urls = [item.get("image") for item in results if item.get("image")]
        img_urls = list(dict.fromkeys(img_urls))  # dedup
        random.shuffle(img_urls)
        return img_urls[:num_images]
    except Exception as e:
        print(f"[ERROR] DuckDuckGo: {e}")
        return []

# --- Download from url list ---
def download_images_from_urls(urls, query, outdir, max_images):
    os.makedirs(outdir, exist_ok=True)
    count = 0
    for i, img_url in enumerate(urls):
        if not img_url:
            continue
        try:
            img_data = requests.get(img_url, timeout=10).content
            fname = os.path.join(outdir, f"{query.replace(' ', '_')}_{count+1}.jpg")
            with open(fname, "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {fname}")
            count += 1
            if count >= max_images:
                break
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] {e}")
    if count == 0:
        print("No images found or download failed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, nargs='*', help='Query หรือหัวข้อภาพ (รองรับหลายหัวข้อ)')
    parser.add_argument('--num_images', type=int, default=5, help='จำนวนภาพต่อหัวข้อ')
    args = parser.parse_args()

    default_queries = [
        "ป้ายประกาศราชการ",
        "ฉลากสินค้าไทย",
        "ใบเสร็จรับเงินภาษาไทย"
    ]
    queries = args.query if args.query else default_queries

    # --- สร้าง output dir ---
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("scraped_images", f"scrape-ocr-{now}")
    img_outdir = os.path.join(outdir, "images")
    os.makedirs(img_outdir, exist_ok=True)
    metadata_path = os.path.join(outdir, "scraped_metadata.jsonl")
    all_metadata = []

    for q in queries:
        print(f"\n[INFO] ดึงภาพ: {q}")
        # รวม url จากทุกแหล่ง (source, url)
        url_sources = []
        for url in bing_image_urls(q, num_images=10):
            url_sources.append((url, "bing"))
        for url in duckduckgo_image_urls(q, num_images=10):
            url_sources.append((url, "duckduckgo"))
        for url in pexels_image_urls(q, num_images=10):
            url_sources.append((url, "pexels"))
        for url in pixabay_image_urls(q, num_images=10):
            url_sources.append((url, "pixabay"))
        # remove duplicates by url
        seen = set()
        url_sources = [x for x in url_sources if not (x[0] in seen or seen.add(x[0]))]
        import random
        random.shuffle(url_sources)
        count = 0
        for i, (img_url, source) in enumerate(url_sources):
            if not img_url:
                continue
            try:
                img_data = requests.get(img_url, timeout=10).content
                fname = f"{q.replace(' ', '_')}_{source}_{count+1}.jpg"
                fpath = os.path.join(img_outdir, fname)
                with open(fpath, "wb") as f:
                    f.write(img_data)
                print(f"Downloaded: {fpath}")
                meta = {
                    "image_path": os.path.relpath(fpath, outdir).replace("\\", "/"),
                    "query": q,
                    "source": source
                }
                all_metadata.append(meta)
                count += 1
                if count >= args.num_images:
                    break
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] {e}")
        if count == 0:
            print(f"No images found or download failed for {q}.")
    # --- export metadata ---
    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"\nExported {len(all_metadata)} metadata entries to {metadata_path}")
    print(f"All images saved to {img_outdir}")

if __name__ == "__main__":
    main()
