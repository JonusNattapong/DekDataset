"""
DekDataset - Data Utils

ชุดเครื่องมือสำหรับการทำความสะอาดข้อมูล การวิเคราะห์ข้อมูล และฟังก์ชันอื่นๆ ที่เกี่ยวข้องกับการจัดการข้อมูล
"""

import re
import json
import html
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from collections import Counter
from typing import List, Dict, Any, Union, Optional
from pythainlp.util import normalize as th_normalize
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# ----------------- Data Cleaning Functions -----------------

def clean_text(text: str, options: Dict = None) -> str:
    """
    ทำความสะอาดข้อความ โดยสามารถเลือกวิธีการทำความสะอาดได้
    
    Args:
        text: ข้อความที่ต้องการทำความสะอาด
        options: ตัวเลือกสำหรับการทำความสะอาด เช่น
            - remove_html: ลบ HTML tags (default: True)
            - remove_urls: ลบ URLs (default: True)
            - remove_emojis: ลบ emojis (default: False)
            - remove_special_chars: ลบอักขระพิเศษ (default: False)
            - normalize_thai: ปรับข้อความภาษาไทยให้เป็นมาตรฐาน (default: True)
            - fix_spacing: แก้ไขช่องว่าง (default: True)
    
    Returns:
        ข้อความที่ทำความสะอาดแล้ว
    """
    if text is None:
        return ""
    
    if options is None:
        options = {}
    
    # กำหนดค่า default ให้กับตัวเลือก
    default_options = {
        "remove_html": True,
        "remove_urls": True,
        "remove_emojis": False,
        "remove_special_chars": False,
        "normalize_thai": True,
        "fix_spacing": True
    }
    
    # รวมตัวเลือกที่กำหนดกับค่า default
    for key, value in default_options.items():
        if key not in options:
            options[key] = value
    
    # ลบ HTML tags
    if options["remove_html"]:
        text = remove_html_tags(text)
    
    # ลบ URLs
    if options["remove_urls"]:
        text = remove_urls(text)
    
    # ลบ emojis
    if options["remove_emojis"]:
        text = remove_emojis(text)
    
    # ลบอักขระพิเศษ
    if options["remove_special_chars"]:
        text = remove_special_chars(text)
    
    # ปรับข้อความภาษาไทยให้เป็นมาตรฐาน
    if options["normalize_thai"]:
        text = normalize_thai_text(text)
    
    # แก้ไขช่องว่าง
    if options["fix_spacing"]:
        text = fix_spacing(text)
    
    return text

def remove_html_tags(text: str) -> str:
    """ลบ HTML tags ออกจากข้อความ"""
    if not text:
        return ""
    
    # ใช้ BeautifulSoup เพื่อแปลง HTML เป็นข้อความธรรมดา
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    
    # แก้ไขอักขระพิเศษของ HTML
    text = html.unescape(text)
    
    return text

def remove_urls(text: str) -> str:
    """ลบ URLs ออกจากข้อความ"""
    if not text:
        return ""
    
    # ลบ URLs ทั่วไป (http, https, www, ftp)
    url_pattern = r'(https?:\/\/|www\.)[^\s]+'
    text = re.sub(url_pattern, ' ', text)
    
    return text

def remove_emojis(text: str) -> str:
    """ลบ emojis ออกจากข้อความ"""
    if not text:
        return ""
    
    # ใช้ emoji library เพื่อลบ emojis
    return emoji.replace_emoji(text, '')

def remove_special_chars(text: str) -> str:
    """ลบอักขระพิเศษออกจากข้อความ ยกเว้นอักขระที่ใช้ในภาษาไทย"""
    if not text:
        return ""
    
    # เก็บเฉพาะตัวอักษรไทย อังกฤษ ตัวเลข และเครื่องหมายที่จำเป็น
    # \u0E00-\u0E7F คือช่วงของอักขระไทย
    pattern = r'[^\u0E00-\u0E7Fa-zA-Z0-9\.\,\!\?\(\)\s]'
    text = re.sub(pattern, ' ', text)
    
    return text

def normalize_thai_text(text: str) -> str:
    """ปรับข้อความภาษาไทยให้เป็นมาตรฐาน"""
    if not text:
        return ""
    
    # ใช้ pythainlp เพื่อปรับข้อความไทยให้เป็นมาตรฐาน
    text = th_normalize(text)
    
    # แปลงเลขอารบิกเป็นเลขไทย (ถ้าต้องการ)
    # thai_digit_map = {"0": "๐", "1": "๑", "2": "๒", "3": "๓", "4": "๔",
    #                  "5": "๕", "6": "๖", "7": "๗", "8": "๘", "9": "๙"}
    # for digit, thai_digit in thai_digit_map.items():
    #     text = text.replace(digit, thai_digit)
    
    return text

def fix_spacing(text: str) -> str:
    """แก้ไขช่องว่างให้เหมาะสม"""
    if not text:
        return ""
    
    # ลบช่องว่างที่มีมากเกินไป
    text = re.sub(r'\s+', ' ', text)
    
    # ตัดช่องว่างที่ขอบ
    text = text.strip()
    
    return text

def normalize_medical_text(text: str) -> str:
    """ทำความสะอาดและปรับข้อความทางการแพทย์ให้เป็นมาตรฐาน"""
    if not text:
        return ""
    
    # ปรับข้อความให้เป็นมาตรฐานก่อน
    text = clean_text(text)
    
    # แปลงคำศัพท์ทางการแพทย์ให้คงที่
    medical_terms_map = {
        # ตัวอย่างการแปลงคำศัพท์ทางการแพทย์
        "ไมเกรน": "ไมเกรน",
        "ไมเกรน": "ไมเกรน",
        "เบาหวาน": "โรคเบาหวาน",
        "diabetes": "โรคเบาหวาน",
        "covid": "โควิด-19",
        "covid19": "โควิด-19",
        "covid-19": "โควิด-19",
        "โควิค": "โควิด-19",
        # เพิ่มเติมตามต้องการ
    }
    
    # ทำการแปลงคำศัพท์
    for term, standard_term in medical_terms_map.items():
        text = re.sub(r'\b' + term + r'\b', standard_term, text, flags=re.IGNORECASE)
    
    return text

def normalize_education_text(text: str) -> str:
    """ทำความสะอาดและปรับข้อความทางการศึกษาให้เป็นมาตรฐาน"""
    if not text:
        return ""
    
    # ปรับข้อความให้เป็นมาตรฐานก่อน
    text = clean_text(text)
    
    # แปลงคำศัพท์ทางการศึกษาให้คงที่
    education_terms_map = {
        # คณิตศาสตร์
        "math": "คณิตศาสตร์",
        "คณิต": "คณิตศาสตร์",
        "เลข": "คณิตศาสตร์",
        # วิทยาศาสตร์
        "science": "วิทยาศาสตร์",
        "วิทย์": "วิทยาศาสตร์",
        "chemistry": "เคมี",
        "physics": "ฟิสิกส์",
        "biology": "ชีววิทยา",
        # ภาษา
        "eng": "ภาษาอังกฤษ",
        "english": "ภาษาอังกฤษ",
        "thai": "ภาษาไทย",
        # ระดับชั้น
        "ป.1": "ประถมศึกษาปีที่ 1",
        "ป.2": "ประถมศึกษาปีที่ 2",
        "ป.3": "ประถมศึกษาปีที่ 3",
        "ป.4": "ประถมศึกษาปีที่ 4",
        "ป.5": "ประถมศึกษาปีที่ 5",
        "ป.6": "ประถมศึกษาปีที่ 6",
        "ม.1": "มัธยมศึกษาปีที่ 1",
        "ม.2": "มัธยมศึกษาปีที่ 2",
        "ม.3": "มัธยมศึกษาปีที่ 3",
        "ม.4": "มัธยมศึกษาปีที่ 4",
        "ม.5": "มัธยมศึกษาปีที่ 5",
        "ม.6": "มัธยมศึกษาปีที่ 6",
    }
    
    # ทำการแปลงคำศัพท์
    for term, standard_term in education_terms_map.items():
        text = re.sub(r'\b' + term + r'\b', standard_term, text, flags=re.IGNORECASE)
    
    return text

def normalize_social_media_text(text: str) -> str:
    """ทำความสะอาดและปรับข้อความจากโซเชียลมีเดียให้เป็นมาตรฐาน"""
    if not text:
        return ""
    
    # ปรับข้อความให้เป็นมาตรฐานก่อน
    text = clean_text(text)
    
    # แปลงคำย่อที่ใช้บ่อยในโซเชียลมีเดีย
    social_media_terms_map = {
        "5555": "ฮาฮาฮา",
        "55+": "ฮาฮาฮา",
        "ชิม": "ชิมิ",
        "มุแงง": "งอนแล้ว",
        "มุงู": "งอนแล้ว",
        "จุงเบย": "จังเลย",
        "จังเบย": "จังเลย",
        "เดะ": "เดี๋ยว",
        "เดรว": "เดี๋ยว",
        "คับ": "ครับ",
        "คะ": "ค่ะ",
        "คร้าบ": "ครับ",
        "ค่ะะ": "ค่ะ",
        "นะค้าบ": "นะครับ",
        "นะค้า": "นะคะ",
    }
    
    # ทำการแปลงคำศัพท์
    for term, standard_term in social_media_terms_map.items():
        text = re.sub(r'\b' + term + r'\b', standard_term, text)
    
    # ลบการซ้ำตัวอักษรที่มากเกินไป (เช่น สวัสดีีีีีีี -> สวัสดี)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    return text

def normalize_ecommerce_text(text: str) -> str:
    """ทำความสะอาดและปรับข้อความจาก E-commerce ให้เป็นมาตรฐาน"""
    if not text:
        return ""
    
    # ปรับข้อความให้เป็นมาตรฐานก่อน
    text = clean_text(text)
    
    # แปลงคำศัพท์ที่ใช้ใน E-commerce ให้คงที่
    ecommerce_terms_map = {
        "ลด": "ลดราคา",
        "ราคาหิ้ว": "ราคานำเข้า",
        "พร้อมส่ง": "พร้อมจัดส่ง",
        "จัดส่งฟรี": "ส่งฟรี",
        "ส่งฟรี ems": "ส่งฟรีแบบลงทะเบียน",
        "พรีออเดอร์": "สั่งล่วงหน้า",
        "preorder": "สั่งล่วงหน้า",
        "พรีออร์เดอร์": "สั่งล่วงหน้า",
        "ของแท้": "สินค้าของแท้",
        "ด่วน": "ด่วน",
        "sale": "ลดราคา",
    }
    
    # ทำการแปลงคำศัพท์
    for term, standard_term in ecommerce_terms_map.items():
        text = re.sub(r'\b' + term + r'\b', standard_term, text, flags=re.IGNORECASE)
    
    # ลบเครื่องหมาย !! หรือ ?? ที่มากเกินไป
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # แปลงตัวเลขเป็นคำอ่าน (เฉพาะบางกรณี)
    # เช่น "1k" -> "หนึ่งพัน", "2.5k" -> "สองพันห้าร้อย"
    text = re.sub(r'(\d+)k\b', lambda m: f"{int(m.group(1))}000", text)
    text = re.sub(r'(\d+\.\d+)k\b', lambda m: f"{float(m.group(1))*1000:.0f}", text)
    
    return text

def remove_duplicate_sentences(text: str) -> str:
    """ลบประโยคที่ซ้ำกันในข้อความ"""
    if not text:
        return ""
    
    # แยกประโยค (ใช้วิธีง่ายๆ โดยแยกตามเครื่องหมาย)
    sentences = re.split(r'([.!?]\s*)', text)
    
    # จัดกลุ่มประโยคกับเครื่องหมาย
    reconstructed_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sentence = sentences[i] + sentences[i+1]
            reconstructed_sentences.append(sentence.strip())
        else:
            reconstructed_sentences.append(sentences[i].strip())
    
    # ลบประโยคที่ซ้ำกัน
    unique_sentences = []
    seen = set()
    for s in reconstructed_sentences:
        normalized_s = s.strip().lower()
        if normalized_s not in seen and normalized_s:
            seen.add(normalized_s)
            unique_sentences.append(s)
    
    # รวมประโยคกลับ
    return ' '.join(unique_sentences)

def correct_common_misspellings(text: str, language: str = "th") -> str:
    """แก้ไขคำผิดที่พบบ่อยตามภาษาที่ระบุ"""
    if not text:
        return ""
    
    if language == "th":
        # คำผิดที่พบบ่อยในภาษาไทย
        corrections = {
            "เเ": "แ",
            "กระเพรา": "กะเพรา",
            "กระทะ": "กะทะ",
            "ผัดไท": "ผัดไทย",
            "นาฬิกา": "นาฬิกา",
            "โทษสัพท์": "โทรศัพท์",
            "โทรสับ": "โทรศัพท์",
            "เสร็จแล้ว": "เสร็จแล้ว",
            "เช็ค": "เช็ก",
        }
    elif language == "en":
        # คำผิดที่พบบ่อยในภาษาอังกฤษ
        corrections = {
            "teh": "the",
            "recieve": "receive",
            "accomodate": "accommodate",
            "occured": "occurred",
            "definately": "definitely",
            "seperate": "separate",
            "wierd": "weird",
            "thier": "their",
        }
    else:
        return text
    
    # แก้ไขคำผิด
    for wrong, correct in corrections.items():
        text = re.sub(r'\b' + wrong + r'\b', correct, text)
    
    return text

# ----------------- Dataset Analysis Functions -----------------

def analyze_dataset(data_entries: List[Dict], field_path: str = "content.text") -> Dict:
    """
    วิเคราะห์ dataset และสร้างสถิติต่างๆ
    
    Args:
        data_entries: รายการข้อมูลในรูปแบบ dictionary
        field_path: path ของฟิลด์ข้อความที่จะวิเคราะห์ (เช่น "content.text")
    
    Returns:
        Dictionary ที่มีผลลัพธ์การวิเคราะห์
    """
    results = {
        "total_entries": len(data_entries),
        "word_stats": {},
        "length_stats": {},
        "language_stats": {},
        "metadata_stats": {},
        "quality_scores": {},
        "sentiment_analysis": {},
        "category_distribution": {},
        "duplicate_stats": {}
    }
    
    texts = []
    lengths = []
    all_words = []
    
    # แยกส่วนของ field_path
    field_parts = field_path.split('.')
    
    # ดึงข้อความจากแต่ละ entry
    for entry in data_entries:
        obj = entry
        for part in field_parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                obj = None
                break
        
        if obj and isinstance(obj, str):
            text = obj
            texts.append(text)
            length = len(text)
            lengths.append(length)
            
            # วิเคราะห์คำ
            words = word_tokenize(text, engine="newmm")
            all_words.extend(words)
    
    # คำนวณสถิติความยาว
    if lengths:
        results["length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
            "std_dev": calculate_stddev(lengths),
            "distribution": calculate_length_distribution(lengths),
            "quantiles": calculate_quantiles(lengths)
        }
    
    # นับความถี่ของคำ
    if all_words:
        stopwords = thai_stopwords()
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 1]
        word_counts = Counter(filtered_words)
        results["word_stats"] = {
            "total_words": len(all_words),
            "unique_words": len(set(all_words)),
            "most_common": word_counts.most_common(20),
            "vocabulary_richness": calculate_vocabulary_richness(all_words),
            "word_length_distribution": calculate_word_length_distribution(all_words)
        }
    
    # วิเคราะห์ Duplicates
    if texts:
        results["duplicate_stats"] = analyze_duplicates(texts)
        
    # วิเคราะห์ Categories/Labels (ถ้ามี)
    category_field = "content.category" if "content.category" in field_path else "category"
    results["category_distribution"] = analyze_categories(data_entries, category_field)
    
    # วิเคราะห์ Metadata
    results["metadata_stats"] = analyze_metadata(data_entries)
    
    return results

def analyze_duplicates(texts: List[str]) -> Dict:
    """วิเคราะห์ข้อความที่ซ้ำกัน"""
    from collections import Counter
    
    normalized_texts = [text.lower().strip() for text in texts]
    duplicates = Counter(normalized_texts)
    
    # ตรวจสอบว่ามีข้อความที่ซ้ำกันหรือไม่
    duplicate_count = sum(1 for count in duplicates.values() if count > 1)
    duplicate_entries = sum(count - 1 for count in duplicates.values() if count > 1)
    
    duplicate_stats = {
        "duplicate_texts": duplicate_count,
        "duplicate_entries": duplicate_entries,
        "unique_texts_percentage": (len(duplicates) / len(texts)) * 100 if texts else 0,
        "duplicate_texts_percentage": (duplicate_count / len(duplicates)) * 100 if duplicates else 0
    }
    
    # หาข้อความที่ซ้ำกันบ่อยที่สุด
    if duplicate_count > 0:
        most_common_duplicates = [
            {"text": text[:50] + "..." if len(text) > 50 else text, "count": count}
            for text, count in duplicates.most_common(5) 
            if count > 1
        ]
        duplicate_stats["most_common_duplicates"] = most_common_duplicates
    
    return duplicate_stats

def calculate_stddev(values: List[float]) -> float:
    """คำนวณค่าเบี่ยงเบนมาตรฐาน"""
    if not values:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def calculate_quantiles(values: List[float]) -> Dict:
    """คำนวณค่า quantiles (25%, 50%, 75%)"""
    if not values:
        return {"q1": 0, "q2": 0, "q3": 0}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    q1_idx = n // 4
    q2_idx = n // 2
    q3_idx = (3 * n) // 4
    
    return {
        "q1": sorted_values[q1_idx],
        "q2": sorted_values[q2_idx],  # same as median
        "q3": sorted_values[q3_idx]
    }

def calculate_length_distribution(lengths: List[int]) -> Dict:
    """คำนวณการกระจายตัวของความยาว แบ่งเป็นช่วง"""
    if not lengths:
        return {}
    
    bins = [
        (0, 50),
        (51, 100),
        (101, 200),
        (201, 500),
        (501, 1000),
        (1001, float('inf'))
    ]
    
    distribution = {f"{start}-{end if end != float('inf') else '∞'}": 0 for start, end in bins}
    
    for length in lengths:
        for start, end in bins:
            if start <= length <= end:
                key = f"{start}-{end if end != float('inf') else '∞'}"
                distribution[key] += 1
                break
    
    # คำนวณเป็นเปอร์เซ็นต์
    total = len(lengths)
    distribution_percent = {k: (v / total) * 100 for k, v in distribution.items()}
    
    return distribution_percent

def calculate_vocabulary_richness(words: List[str]) -> Dict:
    """คำนวณความหลากหลายของคำศัพท์"""
    if not words:
        return {"ttr": 0, "cttr": 0, "rttr": 0}
    
    unique_words = set(words)
    total_words = len(words)
    
    # Type-Token Ratio (TTR)
    ttr = len(unique_words) / total_words
    
    # Corrected Type-Token Ratio (CTTR)
    cttr = len(unique_words) / (total_words ** 0.5)
    
    # Root Type-Token Ratio (RTTR)
    rttr = len(unique_words) / (2 * total_words) ** 0.5
    
    return {
        "ttr": ttr,
        "cttr": cttr,
        "rttr": rttr
    }

def calculate_word_length_distribution(words: List[str]) -> Dict:
    """วิเคราะห์การกระจายตัวของความยาวคำ"""
    if not words:
        return {}
    
    word_lengths = [len(word) for word in words]
    distribution = {}
    
    for length in range(1, max(word_lengths) + 1):
        count = word_lengths.count(length)
        if count > 0:
            distribution[length] = count
    
    return distribution

def analyze_categories(data_entries: List[Dict], category_field: str = "category") -> Dict:
    """วิเคราะห์การกระจายตัวของหมวดหมู่"""
    categories = {}
    
    # แยกส่วนของ field_path
    field_parts = category_field.split('.')
    
    for entry in data_entries:
        obj = entry
        for part in field_parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                obj = None
                break
        
        if obj is not None:
            # แปลงเป็นข้อความเพื่อใช้เป็น key
            category = str(obj)
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
    
    # คำนวณเปอร์เซ็นต์
    total = sum(categories.values())
    categories_percent = {k: (v / total) * 100 for k, v in categories.items()}
    
    return {
        "counts": categories,
        "percentages": categories_percent,
        "total_categories": len(categories),
        "balance_score": calculate_category_balance(categories)
    }

def calculate_category_balance(categories: Dict[str, int]) -> float:
    """คำนวณความสมดุลของหมวดหมู่ (0 = ไม่สมดุลเลย, 1 = สมดุลสมบูรณ์)"""
    if not categories:
        return 1.0
    
    total = sum(categories.values())
    expected = total / len(categories)
    
    # คำนวณค่าเบี่ยงเบนเฉลี่ย
    deviation = sum(abs(count - expected) for count in categories.values()) / total
    
    # แปลงให้อยู่ในช่วง 0-1 โดยที่ 1 คือสมดุลสมบูรณ์
    balance = 1.0 - deviation
    
    return balance

def analyze_metadata(data_entries: List[Dict]) -> Dict:
    """วิเคราะห์ metadata ของ dataset"""
    results = {
        "sources": {},
        "license": {},
        "version": {},
        "language": {},
        "timestamp": {}
    }
    
    for entry in data_entries:
        if "metadata" in entry and isinstance(entry["metadata"], dict):
            metadata = entry["metadata"]
            
            # Source
            if "source" in metadata:
                source = metadata["source"]
                results["sources"][source] = results["sources"].get(source, 0) + 1
            
            # License
            if "license" in metadata:
                license_name = metadata["license"]
                results["license"][license_name] = results["license"].get(license_name, 0) + 1
            
            # Version
            if "version" in metadata:
                version = metadata["version"]
                results["version"][version] = results["version"].get(version, 0) + 1
            
            # Language
            if "lang" in metadata:
                lang = metadata["lang"]
                results["language"][lang] = results["language"].get(lang, 0) + 1
    
    return results

def deduplicate_entries(entries: list, key_fields=None) -> list:
    """Remove duplicate entries by key fields (default: all fields)."""
    def make_hashable(obj):
        if isinstance(obj, (tuple, str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return tuple(make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        else:
            return str(obj)

    seen = set()
    result = []
    for e in entries:
        if key_fields:
            key = tuple(make_hashable(e.get(k)) for k in key_fields)
        else:
            key = tuple(sorted((k, make_hashable(v)) for k, v in e.items()))
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result

# ----------------- Visualization Functions -----------------

def plot_word_cloud(text: str, title: str = "Word Cloud", max_words: int = 200, output_path: str = None):
    """
    สร้าง Word Cloud จากข้อความที่กำหนด
    
    Args:
        text: ข้อความที่ต้องการสร้าง Word Cloud
        title: ชื่อเรื่องที่จะปรากฏบนกราฟ
        max_words: จำนวนคำสูงสุดที่จะปรากฏใน Word Cloud
        output_path: พาธที่จะบันทึกไฟล์ภาพ (ถ้าไม่ระบุจะแสดงบนหน้าจอ)
    """
    from wordcloud import WordCloud
    
    try:
        # สร้าง Word Cloud
        wordcloud = WordCloud(
            background_color='white',
            width=800,
            height=400,
            max_words=max_words,
            random_state=42
        ).generate(text)
        
        # แสดงหรือบันทึกภาพ
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=15)
        
        if output_path:
            plt.savefig(output_path)
            print(f"บันทึกภาพ Word Cloud ไปยัง {output_path}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้าง Word Cloud: {e}")

def plot_category_distribution(categories: Dict, title: str = "Category Distribution", output_path: str = None):
    """
    สร้างกราฟแท่งแสดงการกระจายตัวของหมวดหมู่
    
    Args:
        categories: ข้อมูลหมวดหมู่ที่ต้องการวิเคราะห์
        title: ชื่อเรื่องที่จะปรากฏบนกราฟ
        output_path: พาธที่จะบันทึกไฟล์ภาพ (ถ้าไม่ระบุจะแสดงบนหน้าจอ)
    """
    try:
        category_names = list(categories.keys())
        category_counts = list(categories.values())
        
        # สร้างกราฟแท่ง
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_names, y=category_counts, palette="viridis")
        plt.title(title, fontsize=16)
        plt.xlabel('หมวดหมู่', fontsize=14)
        plt.ylabel('จำนวนตัวอย่าง', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # แสดงค่าเหนือแท่งกราฟ
        for i, count in enumerate(category_counts):
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"บันทึกภาพการกระจายตัวของหมวดหมู่ไปยัง {output_path}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้างกราฟการกระจายตัวของหมวดหมู่: {e}")

def plot_length_distribution(lengths: List[int], title: str = "Length Distribution", output_path: str = None):
    """
    สร้างกราฟแสดงการกระจายตัวของความยาวข้อความ
    
    Args:
        lengths: รายการความยาวของข้อความ
        title: ชื่อเรื่องที่จะปรากฏบนกราฟ
        output_path: พาธที่จะบันทึกไฟล์ภาพ (ถ้าไม่ระบุจะแสดงบนหน้าจอ)
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Histogram
        sns.histplot(lengths, bins=30, kde=True, color='skyblue')
        plt.title(title, fontsize=16)
        plt.xlabel('ความยาวข้อความ (ตัวอักษร)', fontsize=14)
        plt.ylabel('จำนวนตัวอย่าง', fontsize=14)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"บันทึกภาพการกระจายตัวของความยาวข้อความไปยัง {output_path}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้างกราฟการกระจายตัวของความยาวข้อความ: {e}")

def plot_word_frequency(word_counts: List[tuple[str, int]], title: str = "Word Frequency", output_path: str = None):
    """
    สร้างกราฟแท่งแสดงความถี่ของคำที่พบบ่อย
    
    Args:
        word_counts: รายการคำและจำนวนครั้งที่พบ
        title: ชื่อเรื่องที่จะปรากฏบนกราฟ
        output_path: พาธที่จะบันทึกไฟล์ภาพ (ถ้าไม่ระบุจะแสดงบนหน้าจอ)
    """
    try:
        words, counts = zip(*word_counts)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(words), y=list(counts), palette="rocket")
        plt.title(title, fontsize=16)
        plt.xlabel('คำ', fontsize=14)
        plt.ylabel('จำนวนครั้งที่พบ', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # แสดงค่าเหนือแท่งกราฟ
        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"บันทึกภาพความถี่ของคำไปยัง {output_path}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้างกราฟความถี่ของคำ: {e}")

# ----------------- Hugging Face Integration Functions -----------------

def upload_to_huggingface(dataset_path: str, repo_id: str, token: str = None, 
                         private: bool = False, metadata: Dict = None, readme_content: str = None):
    """
    อัพโหลด dataset ไปยัง Hugging Face Hub
    
    Args:
        dataset_path: พาธไปยังโฟลเดอร์ dataset ที่จะอัพโหลด
        repo_id: ชื่อ repository บน Hugging Face Hub (username/repo-name)
        token: Hugging Face API token (ถ้าไม่ระบุจะพยายามใช้จาก environment)
        private: สร้างเป็น repository ส่วนตัวหรือไม่
        metadata: ข้อมูล metadata เพิ่มเติม
        readme_content: เนื้อหาไฟล์ README.md
    
    Returns:
        URL ของ dataset บน Hugging Face Hub
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        
        # สร้าง/ตรวจสอบ repo
        api = HfApi()
        create_repo(repo_id=repo_id, token=token, private=private, repo_type="dataset")
        
        # อัพโหลด
        api.upload_folder(
            folder_path=dataset_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        print(f"อัพโหลด dataset ไปยัง Hugging Face Hub สำเร็จ: https://huggingface.co/datasets/{repo_id}")
        return f"https://huggingface.co/datasets/{repo_id}"
        
    except ImportError:
        print("ไม่พบ huggingface_hub library กรุณาติดตั้งด้วยคำสั่ง 'pip install huggingface_hub'")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอัพโหลดไปยัง Hugging Face Hub: {e}")

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # ตัวอย่างการทำความสะอาดข้อความ
    text = "<p>นี่คือ<b>ตัวอย่าง</b></p> ข้อความ   ที่มี  HTML และช่องว่างผิดปกติ http://example.com"
    cleaned_text = clean_text(text)
    print(f"ข้อความที่ทำความสะอาดแล้ว: {cleaned_text}")
    
    # ตัวอย่างการวิเคราะห์ dataset
    # อ่านไฟล์ jsonl
    try:
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "output")
        sample_file = os.path.join(data_dir, "auto-dataset-sentiment_analysis-20250517-095537.jsonl")
        
        if os.path.exists(sample_file):
            entries = []
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entries.append(json.loads(line))
            
            analysis = analyze_dataset(entries, "content.text")
            print(f"ผลการวิเคราะห์: มีข้อมูล {analysis['total_entries']} รายการ")
            print(f"คำที่พบบ่อย 5 อันดับแรก: {analysis['word_stats'].get('most_common', [])[:5]}")
    except Exception as e:
        print(f"ไม่สามารถวิเคราะห์ข้อมูลตัวอย่างได้: {e}")
    
    # ตัวอย่างการสร้างรายงาน
    try:
        report_dir = os.path.join(data_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # สร้างรายงาน HTML
        create_dataset_report(entries, os.path.join(report_dir, "dataset_report.html"), "content.text")
        
        # สร้างภาพการวิเคราะห์ขั้นสูง
        advanced_visualization(entries, report_dir, "content.text")
    except Exception as e:
        print(f"ไม่สามารถสร้างรายงานได้: {e}")
