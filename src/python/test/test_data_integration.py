#!/usr/bin/env python
# coding: utf-8
"""
DekDataset - Test Data Integration

This script tests the integration between data_utils.py and generate_dataset.py
to ensure that data cleaning, analysis, and Hugging Face Hub integration work as expected.
"""

import os
import json
import argparse
from datetime import datetime
from data_utils import (
    clean_text, analyze_dataset, plot_word_cloud,
    plot_category_distribution, plot_length_distribution,
    plot_word_frequency
)
from colorama import Fore, Style, init

# Initialize colorama
init()

def test_data_cleaning():
    """ทดสอบฟังก์ชันการทำความสะอาดข้อมูลใน data_utils.py"""
    print(f"{Fore.CYAN}[TEST] Testing data cleaning functions...{Style.RESET_ALL}")
    
    test_cases = [
        {
            "input": "<p>นี่คือ<b>ข้อความทดสอบ</b></p> ที่มี  HTML และช่องว่างผิดปกติ http://example.com",
            "expected": "นี่คือข้อความทดสอบ ที่มี HTML และช่องว่างผิดปกติ",
            "options": {"remove_html": True, "remove_urls": True, "fix_spacing": True}
        },
        {
            "input": "ข้อความที่มีเครื่องหมาย emoji 😊 และ 👨‍👩‍👧‍👦 ควรยังคงเก็บไว้",
            "expected": "ข้อความที่มีเครื่องหมาย emoji 😊 และ 👨‍👩‍👧‍👦 ควรยังคงเก็บไว้",
            "options": {"remove_emojis": False}
        },
        {
            "input": "ข้อความที่มีเครื่องหมาย emoji 😊 และ 👨‍👩‍👧‍👦 ควรถูกลบ",
            "expected": "ข้อความที่มีเครื่องหมาย emoji  และ  ควรถูกลบ",
            "options": {"remove_emojis": True}
        },
        {
            "input": "เว็บไซต์ http://example.com และ https://www.test.co.th",
            "expected": "เว็บไซต์  และ ",
            "options": {"remove_urls": True}
        },
        {
            "input": "เว็บไซต์ http://example.com และ https://www.test.co.th",
            "expected": "เว็บไซต์ http://example.com และ https://www.test.co.th",
            "options": {"remove_urls": False}
        },
        {
            "input": "การปรับข้อความภาษาไทย  เเละ  แก้ไขช่องว่าง",
            "expected": "การปรับข้อความภาษาไทย และ แก้ไขช่องว่าง",
            "options": {"normalize_thai": True, "fix_spacing": True}
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases):
        result = clean_text(test["input"], test["options"])
        if result == test["expected"]:
            print(f"{Fore.GREEN}[PASS] Test case {i+1} passed{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[FAIL] Test case {i+1} failed{Style.RESET_ALL}")
            print(f"Expected: {test['expected']}")
            print(f"Got: {result}")
            all_passed = False
    
    if all_passed:
        print(f"{Fore.GREEN}[SUCCESS] All data cleaning tests passed!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[ERROR] Some data cleaning tests failed.{Style.RESET_ALL}")
    
    return all_passed

def test_dataset_analysis():
    """ทดสอบฟังก์ชันการวิเคราะห์ dataset ใน data_utils.py"""
    print(f"{Fore.CYAN}[TEST] Testing dataset analysis functions...{Style.RESET_ALL}")
    
    # สร้าง sample dataset
    sample_data = [
        {
            "id": "test-1",
            "content": {
                "text": "นี่คือข้อความทดสอบสำหรับการวิเคราะห์ dataset ในภาษาไทย"
            },
            "metadata": {"source": "test"}
        },
        {
            "id": "test-2",
            "content": {
                "text": "ข้อความทดสอบที่สอง มีความยาวมากกว่าข้อความแรกเล็กน้อย และมีคำซ้ำๆ เช่น ทดสอบ ทดสอบ"
            },
            "metadata": {"source": "test"}
        },
        {
            "id": "test-3",
            "content": {
                "text": "ข้อความที่สาม"
            },
            "metadata": {"source": "test"}
        }
    ]
    
    # ทดสอบการวิเคราะห์
    try:
        analysis = analyze_dataset(sample_data, field_path="content.text")
        print(f"{Fore.GREEN}[SUCCESS] Dataset analysis successful!{Style.RESET_ALL}")
        print(f"Total entries: {analysis['total_entries']}")
        print(f"Total words: {analysis['word_stats']['total_words']}")
        print(f"Unique words: {analysis['word_stats']['unique_words']}")
        print(f"Average text length: {analysis['length_stats']['avg']:.2f} characters")
        
        # ทดสอบการสร้างภาพแสดงผล
        try:
            test_output_dir = "data/output/test"
            os.makedirs(test_output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            base_path = os.path.join(test_output_dir, f"test_viz_{timestamp}")
            
            # Generate different visualizations
            word_cloud_path = f"{base_path}_wordcloud.png"
            plot_word_cloud(" ".join([entry["content"]["text"] for entry in sample_data]), output_path=word_cloud_path)
            
            if any("category" in entry["content"] for entry in sample_data):
                category_path = f"{base_path}_categories.png"
                categories = {entry["content"]["category"]: 1 for entry in sample_data}
                plot_category_distribution(categories, output_path=category_path)
            
            length_path = f"{base_path}_lengths.png"
            lengths = [len(entry["content"]["text"]) for entry in sample_data]
            plot_length_distribution(lengths, output_path=length_path)
            
            freq_path = f"{base_path}_word_freq.png"
            all_text = " ".join([entry["content"]["text"] for entry in sample_data])
            from collections import Counter
            word_counts = Counter(all_text.split()).most_common(20)
            plot_word_frequency(word_counts, output_path=freq_path)
            
            print(f"{Fore.GREEN}[SUCCESS] Dataset visualizations created in {test_output_dir}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Dataset visualization failed: {e}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Dataset analysis failed: {e}{Style.RESET_ALL}")
        return False

def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description="DekDataset - Test Data Integration")
    parser.add_argument("--test-cleaning", action="store_true", help="Test data cleaning functions")
    parser.add_argument("--test-analysis", action="store_true", help="Test dataset analysis functions")
    parser.add_argument("--test-all", action="store_true", help="Test all functions")
    args = parser.parse_args()
    
    if args.test_all or args.test_cleaning:
        test_data_cleaning()
        
    if args.test_all or args.test_analysis:
        test_dataset_analysis()
        
    if not (args.test_all or args.test_cleaning or args.test_analysis):
        print("No tests specified. Use --test-all, --test-cleaning, or --test-analysis")
        parser.print_help()

if __name__ == "__main__":
    print(f"{Fore.CYAN}=== DekDataset - Data Integration Tests ==={Style.RESET_ALL}")
    main()
