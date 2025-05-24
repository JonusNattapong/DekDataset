import glob
import json

# รวมไฟล์ dataset ป.1-ม.6 ที่เกี่ยวข้อง
import glob

# ค้นหาไฟล์ทั้งหมดที่เป็น dataset ที่เกี่ยวกับ primary และ secondary
input_files = [
    'data/output/auto-dataset-primary-exam-sample.jsonl',
    'data/output/auto-dataset-secondary-exam-sample.jsonl',
    'data/output/complete-diverse-dataset.jsonl',  # dataset ที่สร้างใหม่
]

# เพิ่มไฟล์อื่น ๆ ที่มีในระบบ
additional_files = glob.glob('data/output/auto-dataset-*-*.jsonl')
for file in additional_files:
    if file not in input_files and not file.endswith("sentiment_analysis") and not file.endswith("summarization") and not file.endswith("translation"):
        input_files.append(file)

print(f"รวมไฟล์ทั้งหมด {len(input_files)} ไฟล์: {input_files}")
output_file = 'data/output/complete-merged-dataset.jsonl'

with open(output_file, 'w', encoding='utf-8') as fout:
    for fname in input_files:
        with open(fname, encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    fout.write(line)

print(f"Merged {len(input_files)} files -> {output_file}")
