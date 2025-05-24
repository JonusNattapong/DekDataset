import os
import requests
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
import json
import argparse
from typing import List
from pdf2image import convert_from_path
import tempfile

# Optional: For Wikimedia Commons and Wikipedia
import wikipedia
import urllib.parse

# --- GOT-OCR2_0 Pipeline ---
def run_got_ocr2_on_images(image_paths: List[str], model_name="stepfun-ai/GOT-OCR2_0"):
    ocr = pipeline("image-to-text", model=model_name, trust_remote_code=True)
    results = []
    for img_path in tqdm(image_paths, desc="OCR"):
        try:
            result = ocr(img_path)
            results.append({"image": img_path, "ocr": result})
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
    return results

def run_smol_docling_ocr_on_images(image_paths_or_urls, model_name="ds4sd/SmolDocling-256M-preview", is_url=False):
    pipe = pipeline("image-text-to-text", model=model_name)
    results = []
    for img in tqdm(image_paths_or_urls, desc="OCR-SmolDocling"):
        try:
            if is_url:
                image_input = {"type": "image", "url": img}
            else:
                image_input = {"type": "image", "path": img}
            messages = [
                {
                    "role": "user",
                    "content": [
                        image_input,
                        {"type": "text", "text": "What is written in this image?"}
                    ]
                }
            ]
            result = pipe(messages)
            results.append({"image": img, "ocr": result})
        except Exception as e:
            print(f"[ERROR] {img}: {e}")
    return results

# --- Wikipedia Context Fetcher ---
def get_wikipedia_summary(title, lang="en"):
    wikipedia.set_lang(lang)
    try:
        return wikipedia.summary(title)
    except Exception as e:
        print(f"[WARN] Wikipedia: {e}")
        return ""

def convert_documents_to_images(input_dir, out_img_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    image_paths = []
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        ext = fname.lower().split('.')[-1]
        if ext in ["pdf"]:
            try:
                pages = convert_from_path(fpath)
                for i, page in enumerate(pages):
                    img_path = os.path.join(out_img_dir, f"{os.path.splitext(fname)[0]}_page{i+1}.png")
                    page.save(img_path, "PNG")
                    image_paths.append(img_path)
            except Exception as e:
                print(f"[WARN] PDF to image failed: {fname}: {e}")
        elif ext in ["docx"]:
            # Optional: implement DOCX to image if needed
            print(f"[WARN] DOCX to image not implemented: {fname}")
    return image_paths

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, required=True, help="OCR all images/docs in this local directory (jpg/png/pdf)")
    parser.add_argument("--outdir", type=str, default="ocr_images")
    parser.add_argument("--wikipedia", type=str, default=None, help="Wikipedia title for context")
    args = parser.parse_args()

    print(f"[INFO] Using local files from: {args.local_dir}")
    # 1. Convert PDF (and optionally DOCX) to images
    temp_img_dir = os.path.join(args.outdir, "_converted_images")
    doc_imgs = convert_documents_to_images(args.local_dir, temp_img_dir)
    # 2. Collect all images (original + converted)
    image_paths = [os.path.join(args.local_dir, f) for f in os.listdir(args.local_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths += doc_imgs
    print(f"[INFO] Found {len(image_paths)} images (including converted from docs)")
    print(f"[INFO] Run SmolDocling OCR on images...")
    ocr_results = run_smol_docling_ocr_on_images(image_paths)

    # Optionally fetch Wikipedia context
    context = None
    if args.wikipedia:
        context = get_wikipedia_summary(args.wikipedia)
        print(f"[INFO] Wikipedia context: {context[:200]}...")

    # Export results
    out_json = os.path.join(args.outdir, "ocr_results.jsonl")
    with open(out_json, "w", encoding="utf-8") as f:
        for r in ocr_results:
            entry = {"image": r["image"], "ocr": r["ocr"]}
            if context:
                entry["context"] = context
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Exported OCR results to {out_json}")

if __name__ == "__main__":
    main()
