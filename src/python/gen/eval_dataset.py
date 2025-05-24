import json
import sys
from typing import List, Dict

def load_dataset(path: str) -> List[Dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_answers(path: str) -> Dict:
    """สมมติ answer file เป็น jsonl: {"id":..., "answer":...} หรือ {"question":..., "answer":...}"""
    answers = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                key = obj.get('id') or obj.get('question')
                answers[key] = obj.get('answer')
    return answers

def evaluate(dataset_path: str, answer_path: str):
    data = load_dataset(dataset_path)
    answers = load_answers(answer_path)
    total = 0
    correct = 0
    for item in data:
        key = item.get('id') or item.get('question')
        pred = item.get('prediction')
        gt = answers.get(key)
        if pred is not None and gt is not None:
            total += 1
            if str(pred).strip() == str(gt).strip():
                correct += 1
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    if total:
        print(f"Accuracy: {correct/total:.2%}")
    else:
        print("No matched predictions to evaluate.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_dataset.py <dataset_with_prediction.jsonl> <answer_file.jsonl>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
