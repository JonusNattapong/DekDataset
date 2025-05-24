import csv
import random
import os
import json

def load_tatoeba_sentences(sent_path, lang):
    """Load sentences.csv, filter by language code."""
    sents = {}
    with open(sent_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) < 3:
                continue
            sid, l, text = row
            if l == lang:
                sents[sid] = text
    return sents

def load_tatoeba_links(link_path):
    """Load links.csv, return list of (sid1, sid2)."""
    links = []
    with open(link_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) < 2:
                continue
            links.append((row[0], row[1]))
    return links

def build_parallel_dataset(sent1, sent2, links, max_pairs=1000):
    """Build parallel pairs from links and two sentence dicts."""
    pairs = []
    for sid1, sid2 in links:
        if sid1 in sent1 and sid2 in sent2:
            pairs.append({"source": sent1[sid1], "target": sent2[sid2]})
        elif sid2 in sent1 and sid1 in sent2:
            pairs.append({"source": sent1[sid2], "target": sent2[sid1]})
    random.shuffle(pairs)
    return pairs[:max_pairs]

def main():
    # Path to Tatoeba files (download from https://downloads.tatoeba.org/exports/)
    sent_path = "sentences.csv"  # or full path
    link_path = "links.csv"
    src_lang = "en"
    tgt_lang = "th"
    out_path = f"tatoeba_{src_lang}_{tgt_lang}_1000.jsonl"
    max_pairs = 1000

    print(f"Loading sentences for {src_lang} and {tgt_lang}...")
    sents_src = load_tatoeba_sentences(sent_path, src_lang)
    sents_tgt = load_tatoeba_sentences(sent_path, tgt_lang)
    print(f"Loaded {len(sents_src)} {src_lang}, {len(sents_tgt)} {tgt_lang} sentences.")
    print("Loading links...")
    links = load_tatoeba_links(link_path)
    print(f"Loaded {len(links)} links.")
    print("Building parallel pairs...")
    pairs = build_parallel_dataset(sents_src, sents_tgt, links, max_pairs)
    print(f"Writing {len(pairs)} pairs to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
