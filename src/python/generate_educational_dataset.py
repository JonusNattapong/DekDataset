import os
import json
import re
import pandas as pd
from typing import List, Dict, Any, Union
from ocr_utils import extract_educational_content
from rag_utils import synthesize_educational_content

def generate_qa_pairs(content: Dict[str, Any], num_pairs: int = 50) -> List[Dict[str, Any]]:
    """Generate Q&A pairs from educational content with error handling"""
    qa_pairs = []
    
    # Handle sections
    for section in content.get('sections', []):
        title = section.get('title', '')
        text = section.get('content', '')
        
        if not title or not text:
            continue
            
        # Basic comprehension questions
        qa_pairs.append({
            "question": f"อธิบายความหมายของ{title}",
            "answer": text[:500],  # First 500 chars as summary
            "type": "comprehension",
            "topic": title
        })
        
        # Topic-specific questions
        if 'เศรษฐกิจพอเพียง' in title:
            qa_pairs.append({
                "question": "หลักการสำคัญของเศรษฐกิจพอเพียงมีอะไรบ้าง",
                "answer": text,
                "type": "concept",
                "topic": "เศรษฐกิจพอเพียง"
            })
        
        # Generate from section examples if available
        for example in section.get('examples', []):
            if example:
                qa_pairs.append({
                    "question": "จงยกตัวอย่างการประยุกต์ใช้เศรษฐกิจพอเพียง",
                    "answer": example,
                    "type": "example",
                    "topic": "การประยุกต์ใช้"
                })
    
    # Generate from key concepts if available
    for concept in content.get('key_concepts', []):
        if concept:
            qa_pairs.append({
                "question": "อธิบายแนวคิดสำคัญของเศรษฐกิจพอเพียง",
                "answer": concept,
                "type": "concept",
                "topic": "แนวคิดหลัก"
            })
    
    # Generate from learning objectives if available
    for objective in content.get('learning_objectives', []):
        if objective:
            qa_pairs.append({
                "question": "จุดประสงค์การเรียนรู้ของเศรษฐกิจพอเพียงคืออะไร",
                "answer": objective,
                "type": "objective",
                "topic": "จุดประสงค์การเรียนรู้"
            })
    
    # Ensure we have at least one QA pair
    if not qa_pairs:
        print("[WARN] No QA pairs could be generated from content")
        qa_pairs.append({
            "question": "อธิบายความหมายของเศรษฐกิจพอเพียง",
            "answer": "ไม่พบเนื้อหาที่เกี่ยวข้อง",
            "type": "comprehension",
            "topic": "เศรษฐกิจพอเพียง"
        })
    
    return qa_pairs[:num_pairs]

def extract_metadata(content: Dict[str, Any]) -> Dict[str, str]:
    """Extract metadata from content dynamically"""
    # Try to extract title from first section or content
    title = ""
    if content.get('sections'):
        first_section = content['sections'][0]
        if 'title' in first_section:
            title = first_section['title'].split()[0]  # Get first word as subject
            
    # Try to extract level from content
    level = ""
    for section in content.get('sections', []):
        text = section.get('content', '').lower()
        if 'มัธยมศึกษา' in text:
            level = 'มัธยมศึกษา'
            break
        elif 'ประถมศึกษา' in text:
            level = 'ประถมศึกษา'
            break
            
    # Try to extract subject
    subject = ""
    for section in content.get('sections', []):
        text = section.get('content', '').lower()
        if 'สาระ' in text and 'รายวิชา' in text:
            # Extract text between สาระ and รายวิชา
            try:
                subject = re.search(r'สาระ(.*?)รายวิชา', text).group(1).strip()
            except:
                pass
            break
    
    return {
        'title': title,
        'level': level, 
        'subject': subject
    }

def structure_content(raw_content: Dict[str, Any], enhanced_content: Dict[str, Any]) -> Dict[str, Any]:
    """Structure content with dynamic metadata"""
    
    # Extract metadata dynamically
    metadata = extract_metadata(raw_content)
    
    structured = {
        # Dynamic metadata
        'title': metadata['title'],
        'level': metadata['level'],
        'subject': metadata['subject'],
        
        # Key concepts and objectives from enhanced content
        'key_concepts': enhanced_content.get('key_concepts', []),
        'learning_objectives': enhanced_content.get('learning_objectives', []),
        
        'sections': []
    }
    
    # Process sections
    for raw_section, enhanced_section in zip(
        raw_content.get('sections', []),
        enhanced_content.get('sections', [])
    ):
        section = {
            'title': raw_section.get('title', ''),
            'content': raw_section.get('content', ''),
            'key_points': enhanced_section.get('key_points', []),
            'examples': enhanced_section.get('examples', []),
            'summary': enhanced_section.get('summary', '')
        }
        structured['sections'].append(section)
    
    return structured

def convert_to_csv(structured_content: Dict[str, Any], output_path: str):
    """Convert structured content to CSV with dynamic columns"""
    
    # Get all possible keys from content
    keys = set()
    for section in structured_content.get('sections', []):
        keys.update(section.keys())
    
    # Prepare rows with dynamic fields
    rows = []
    
    # Add metadata if exists
    metadata_fields = ['title', 'level', 'subject']
    if any(structured_content.get(f) for f in metadata_fields):
        row = {'type': 'metadata'}
        for field in metadata_fields:
            row[field] = structured_content.get(field, '')
        rows.append(row)
    
    # Add concepts if exists
    for concept in structured_content.get('key_concepts', []):
        rows.append({
            'type': 'key_concept',
            'content': concept
        })
        
    # Add objectives if exists  
    for objective in structured_content.get('learning_objectives', []):
        rows.append({
            'type': 'learning_objective',
            'content': objective
        })
    
    # Add sections with all available fields
    for section in structured_content.get('sections', []):
        row = {'type': 'section'}
        for key in keys:
            if key in section:
                # Join lists with newlines
                value = section[key]
                if isinstance(value, list):
                    value = '\n'.join(value)
                row[key] = value
        rows.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')

def load_existing_content(filepath: str) -> Dict[str, Any]:
    """Load existing content from file"""
    try:
        if filepath.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        else:
            print(f"[WARN] Unsupported file format: {filepath}")
            return None
    except Exception as e:
        print(f"[WARN] Failed to load existing content: {e}")
        return None

def main():
    pdf_path = r"D:\Github\DekDataset\data\pdf\SallySilpatham-SawasdiBook\sawasdi1.pdf"
    output_dir = r"D:\Github\DekDataset\data\output"
    
    raw_jsonl_path = os.path.join(output_dir, "sufficiency_economy_raw.jsonl")
    raw_csv_path = os.path.join(output_dir, "sufficiency_economy_raw.csv")
    
    # Check for existing content
    raw_content = None
    if os.path.exists(raw_jsonl_path):
        print("[INFO] Found existing JSONL content, loading...")
        raw_content = load_existing_content(raw_jsonl_path)
    elif os.path.exists(raw_csv_path):
        print("[INFO] Found existing CSV content, loading...")
        raw_content = load_existing_content(raw_csv_path)
    
    if not raw_content:
        # Extract new content from PDF
        print("[INFO] No existing content found, extracting from PDF...")
        raw_content = extract_educational_content(pdf_path)
        
        if not raw_content:
            print("[ERROR] Failed to extract content")
            return
    
    # Continue with RAG enhancement
    print("[INFO] Enhancing content with RAG...")
    enhanced_content = synthesize_educational_content(raw_content)
    
    # Structure the content
    print("[INFO] Structuring content...")
    structured_content = structure_content(raw_content, enhanced_content)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sufficiency_economy_raw.csv")
    convert_to_csv(structured_content, csv_path)
    print(f"[INFO] Saved structured content to {csv_path}")
    
    # Generate and save QA pairs
    print("[INFO] Generating QA pairs...")
    qa_pairs = generate_qa_pairs(structured_content, num_pairs=20)
    qa_path = os.path.join(output_dir, "sufficiency_economy_qa.jsonl")
    with open(qa_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"[INFO] Saved {len(qa_pairs)} QA pairs to {qa_path}")

if __name__ == "__main__":
    main()
