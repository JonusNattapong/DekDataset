# Dataset Analysis Tools

DekDataset มีเครื่องมือสำหรับการวิเคราะห์ชุดข้อมูลเพื่อให้เข้าใจลักษณะและคุณภาพของข้อมูล

## ฟังก์ชันวิเคราะห์ข้อมูล

### การวิเคราะห์พื้นฐาน

- **analyze_dataset()** - วิเคราะห์ dataset และสร้างสถิติต่างๆ เช่น จำนวนคำ, ความยาวข้อความ, คำที่พบบ่อย
- **visualize_dataset()** - สร้างภาพแสดงการวิเคราะห์ข้อมูลของ dataset เช่น histogram, boxplot, bar chart

## วิธีการใช้งาน

### การวิเคราะห์ Dataset

```python
from data_utils import analyze_dataset
import json

# อ่านข้อมูลจากไฟล์ JSONL
data_entries = []
with open("data/output/my-dataset.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data_entries.append(json.loads(line))

# วิเคราะห์ข้อมูล (สำหรับข้อมูลที่มีโครงสร้าง content.text)
analysis = analyze_dataset(data_entries, field_path="content.text")

# แสดงสถิติ
print(f"จำนวนตัวอย่างทั้งหมด: {analysis['total_entries']}")
print(f"จำนวนคำทั้งหมด: {analysis['word_stats']['total_words']}")
print(f"จำนวนคำที่ไม่ซ้ำ: {analysis['word_stats']['unique_words']}")
print(f"ความยาวข้อความเฉลี่ย: {analysis['length_stats']['avg']:.2f} ตัวอักษร")
print(f"ความยาวข้อความต่ำสุด: {analysis['length_stats']['min']} ตัวอักษร")
print(f"ความยาวข้อความสูงสุด: {analysis['length_stats']['max']} ตัวอักษร")
print(f"คำที่พบบ่อย 10 อันดับแรก: {analysis['word_stats'].get('most_common', [])[:10]}")
```

### การสร้างภาพแสดงผล

```python
from data_utils import visualize_dataset

# สร้างภาพแสดงผล
visualize_dataset(data_entries, field_path="content.text", output_path="data/output/dataset_analysis.png")
```

### การใช้เครื่องมือบรรทัดคำสั่ง

สามารถใช้งานฟังก์ชันการวิเคราะห์ผ่านคำสั่ง `generate_dataset.py` ได้ดังนี้:

```bash
# วิเคราะห์ dataset และบันทึกผลเป็นไฟล์ JSON
python src/python/generate_dataset.py sentiment_analysis 100 --analyze

# วิเคราะห์และสร้างภาพแสดงผล
python src/python/generate_dataset.py sentiment_analysis 100 --analyze --visualize

# กำหนดตำแหน่งบันทึกผลการวิเคราะห์
python src/python/generate_dataset.py sentiment_analysis 100 --analyze --analyze-output data/analysis/my-analysis
```

## ตัวอย่างผลการวิเคราะห์

การวิเคราะห์ dataset จะให้ข้อมูลสำคัญเกี่ยวกับชุดข้อมูลของคุณ เช่น:

1. **สถิติความยาวข้อความ**: ช่วยให้ทราบว่าข้อความในชุดข้อมูลมีความยาวเหมาะสมหรือไม่
2. **จำนวนและความถี่ของคำ**: ช่วยให้เข้าใจคำศัพท์ที่ใช้ในชุดข้อมูล
3. **การกระจายตัว**: แสดงให้เห็นความสมดุลของข้อมูล

การวิเคราะห์เหล่านี้สามารถช่วยในการตัดสินใจว่าจำเป็นต้องทำการปรับปรุงคุณภาพข้อมูลหรือไม่ เช่น:

- กรองข้อความที่สั้นหรือยาวเกินไป
- เพิ่มความหลากหลายของคำศัพท์
- ปรับความสมดุลของหมวดหมู่หรือป้ายกำกับ
