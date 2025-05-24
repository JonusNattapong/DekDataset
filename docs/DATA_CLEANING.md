# Data Cleaning & Text Normalization

DekDataset มีฟังก์ชันสำหรับการทำความสะอาดข้อมูลและการ normalize ข้อความภาษาไทย เพื่อให้ได้ชุดข้อมูลที่มีคุณภาพสูงสำหรับการเทรนโมเดล AI

## ฟังก์ชันที่มีให้ใช้งาน

### การทำความสะอาดข้อความ

- **clean_text()** - ทำความสะอาดข้อความโดยรวม สามารถกำหนดตัวเลือกได้
- **remove_html_tags()** - ลบ HTML tags ออกจากข้อความ
- **remove_urls()** - ลบ URLs ออกจากข้อความ
- **remove_emojis()** - ลบ emojis ออกจากข้อความ
- **remove_special_chars()** - ลบอักขระพิเศษออกจากข้อความ (ยกเว้นอักขระที่ใช้ในภาษาไทย)
- **normalize_thai_text()** - ปรับข้อความภาษาไทยให้เป็นมาตรฐาน
- **fix_spacing()** - แก้ไขช่องว่างให้เหมาะสม

### การ Normalize เฉพาะด้าน

- **normalize_medical_text()** - ทำความสะอาดและปรับข้อความทางการแพทย์ให้เป็นมาตรฐาน

## วิธีการใช้งาน

### การทำความสะอาดข้อความพื้นฐาน

```python
from data_utils import clean_text

# ทำความสะอาดข้อความแบบง่าย (ใช้ค่า default)
text = "<p>นี่คือ<b>ตัวอย่าง</b></p> ข้อความ   ที่มี  HTML และช่องว่างผิดปกติ http://example.com"
cleaned_text = clean_text(text)
print(cleaned_text)  # นี่คือตัวอย่าง ข้อความ ที่มี HTML และช่องว่างผิดปกติ
```

### การกำหนดตัวเลือกเพิ่มเติม

```python
# กำหนดตัวเลือกเพิ่มเติม
options = {
    "remove_html": True,
    "remove_urls": True, 
    "remove_emojis": True,  # ลบ emoji ออก
    "remove_special_chars": True,  # ลบอักขระพิเศษ
    "normalize_thai": True,
    "fix_spacing": True
}

text = "🙂 นี่คือตัวอย่าง URL: https://example.com และมีอักขระ@พิเศษ#$%^"
cleaned_text = clean_text(text, options)
print(cleaned_text)  # นี่คือตัวอย่าง URL และมีอักขระพิเศษ
```

### การใช้เครื่องมือบรรทัดคำสั่ง

สามารถใช้งานฟังก์ชันการทำความสะอาดข้อมูลผ่านคำสั่ง `generate_dataset.py` ได้ดังนี้:

```bash
python src/python/generate_dataset.py sentiment_analysis 100 --no-clean  # ปิดการทำความสะอาดข้อมูล
python src/python/generate_dataset.py sentiment_analysis 100 --disable-thai-norm  # ปิดการ normalize ภาษาไทย
python src/python/generate_dataset.py sentiment_analysis 100 --remove-emojis  # ลบ emojis
python src/python/generate_dataset.py sentiment_analysis 100 --remove-special-chars  # ลบอักขระพิเศษ
```

## ข้อมูลเพิ่มเติม

การทำความสะอาดข้อมูลเป็นส่วนสำคัญของการเตรียมข้อมูลสำหรับการเทรนโมเดล NLP ฟังก์ชันใน DekDataset ถูกออกแบบมาเพื่อให้มีความยืดหยุ่น สามารถปรับแต่งตามความต้องการได้ และเหมาะสมกับข้อความภาษาไทย
