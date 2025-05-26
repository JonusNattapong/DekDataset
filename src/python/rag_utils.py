import os
import json
import time
import requests
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

# Cache for RAG results
_rag_cache = {}

def generate_rag_content(context: str, instruction: str, timeout: int = 30) -> str:
    """Generate content using DeepSeek API with RAG and timeout"""
    
    # Check cache first
    cache_key = f"{hash(context)}-{hash(instruction)}"
    if cache_key in _rag_cache:
        return _rag_cache[cache_key]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are an educational content expert specializing in Thai curriculum."
            },
            {
                "role": "user", 
                "content": f"Given this context:\n\n{context}\n\nInstruction: {instruction}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # Cache the result
        _rag_cache[cache_key] = result
        return result
        
    except requests.Timeout:
        print(f"[WARN] DeepSeek API timeout after {timeout}s")
        return ""
    except Exception as e:
        print(f"[ERROR] DeepSeek API call failed: {e}")
        return ""

def synthesize_educational_content(
    raw_content: Dict[str, Any],
    batch_size: int = 3,
    timeout: int = 30
) -> Dict[str, Any]:
    """Synthesize educational content using RAG with batching"""
    
    enhanced_content = {
        "sections": [],
        "examples": [],
        "exercises": [],
        "key_concepts": [],
        "learning_objectives": []
    }
    
    print("[INFO] Generating learning objectives...")
    context = json.dumps(raw_content, ensure_ascii=False)
    objectives = generate_rag_content(
        context,
        "Generate 5-7 clear learning objectives for this content in Thai language.",
        timeout=timeout
    )
    enhanced_content["learning_objectives"] = objectives.split("\n")
    
    # Process sections in batches
    sections = raw_content.get("sections", [])
    batches = [sections[i:i+batch_size] for i in range(0, len(sections), batch_size)]
    
    print(f"[INFO] Processing {len(sections)} sections in {len(batches)} batches...")
    for batch_num, batch in enumerate(batches, 1):
        print(f"[INFO] Processing batch {batch_num}/{len(batches)}...")
        
        for section in tqdm(batch, desc="Sections"):
            enhanced_section = {
                "title": section["title"],
                "content": section["content"],
                "key_points": [],
                "examples": []
            }
            
            # Generate key points
            key_points = generate_rag_content(
                section["content"],
                "Extract 3-5 key points from this section in Thai language.",
                timeout=timeout
            )
            enhanced_section["key_points"] = key_points.split("\n")
            
            # Generate examples
            examples = generate_rag_content(
                section["content"],
                "Generate 2-3 practical examples applying these concepts in Thai language.",
                timeout=timeout  
            )
            enhanced_section["examples"] = examples.split("\n")
            
            enhanced_content["sections"].append(enhanced_section)
            
            # Add delay between sections to avoid rate limits
            time.sleep(1)
            
        # Add delay between batches
        if batch_num < len(batches):
            print("[INFO] Batch complete, waiting before next batch...")
            time.sleep(3)
    
    return enhanced_content

def generate_qa_pairs(content: Dict[str, Any], num_pairs: int = 20) -> List[Dict[str, Any]]:
    """Generate diverse Q&A pairs from educational content with better coverage"""
    qa_pairs = []
    
    # Question types and templates
    comprehension_templates = [
        "อธิบายความหมายของ{topic}",
        "{topic}มีประโยชน์อย่างไร",
        "อธิบายหลักการสำคัญของ{topic}",
        "{topic}มีองค์ประกอบอะไรบ้าง"
    ]
    
    concept_templates = [
        "หลักการสำคัญของ{topic}มีอะไรบ้าง",
        "{topic}มีแนวทางการปฏิบัติอย่างไร",
        "จงยกตัวอย่างการประยุกต์ใช้{topic}",
        "เพราะเหตุใด{topic}จึงมีความสำคัญ"
    ]
    
    # Generate from sections with diverse question types
    for section in content['sections']:
        title = section['title']
        text = section['content']
        
        if not text or len(text) < 50:
            continue
            
        # Basic comprehension question
        qa_pairs.append({
            "question": random.choice(comprehension_templates).format(topic=title),
            "answer": text[:500],
            "type": "comprehension",
            "topic": title
        })
        
        # Topic-specific questions for key concepts
        if any(x in title.lower() for x in ['เศรษฐกิจพอเพียง', 'หลักปรัชญา', 'การพัฒนา']):
            qa_pairs.append({
                "question": random.choice(concept_templates).format(topic=title),
                "answer": text,
                "type": "concept",
                "topic": title
            })
    
    # Generate from examples
    for example in content.get('examples', []):
        if not example or len(example) < 50:
            continue
            
        qa_pairs.append({
            "question": "จงยกตัวอย่างการประยุกต์ใช้เศรษฐกิจพอเพียง",
            "answer": example,
            "type": "example",
            "topic": "การประยุกต์ใช้"
        })
    
    # Generate from exercises
    for exercise in content.get('exercises', []):
        if not exercise or len(exercise) < 10:
            continue
            
        qa_pairs.append({
            "question": exercise,
            "answer": "",  # Leave empty for manual filling
            "type": "exercise", 
            "topic": "แบบฝึกหัด"
        })
    
    # Filter duplicates and limit results
    seen = set()
    filtered_pairs = []
    
    for pair in qa_pairs:
        key = f"{pair['question']}-{pair['type']}"
        if key not in seen:
            seen.add(key)
            filtered_pairs.append(pair)
            
    # Ensure good mix of question types
    final_pairs = []
    type_counts = {'comprehension': 0, 'concept': 0, 'example': 0, 'exercise': 0}
    target_per_type = num_pairs // 4
    
    for pair in filtered_pairs:
        qtype = pair['type']
        if type_counts[qtype] < target_per_type:
            final_pairs.append(pair)
            type_counts[qtype] += 1
            
        if len(final_pairs) >= num_pairs:
            break
            
    return final_pairs
