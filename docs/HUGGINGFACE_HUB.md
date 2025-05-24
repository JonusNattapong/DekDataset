# Hugging Face Hub Integration

DekDataset มีฟังก์ชันสำหรับการอัปโหลดชุดข้อมูลไปยัง Hugging Face Hub เพื่อให้สามารถแชร์และใช้งานชุดข้อมูลได้ง่าย

## ฟังก์ชันที่มีให้ใช้งาน

- **upload_to_huggingface()** - อัปโหลดชุดข้อมูลไปยัง Hugging Face Hub

## ข้อกำหนดเบื้องต้น

1. มี Hugging Face Account
2. สร้าง API Token จาก [Hugging Face Settings](https://huggingface.co/settings/tokens) (ต้องมีสิทธิ์ `write`)
3. ติดตั้ง huggingface_hub package: `pip install huggingface_hub`

## วิธีการใช้งาน

### การอัปโหลดชุดข้อมูลด้วย Python

```python
from data_utils import upload_to_huggingface

# อัปโหลดชุดข้อมูลไปยัง Hugging Face Hub
dataset_url = upload_to_huggingface(
    dataset_path="data/output/my-dataset-folder",  # โฟลเดอร์ที่มีไฟล์ข้อมูล
    repo_id="username/dataset-name",               # ชื่อ repository (username/repo-name)
    token="hf_...",                                # Hugging Face API token
    private=False,                                 # กำหนดเป็น public หรือ private
    metadata={                                     # ข้อมูล metadata เพิ่มเติม (optional)
        "language": "th",
        "license": "CC-BY-4.0",
        "source": "DekDataset" 
    },
    readme_content=None                            # เนื้อหาไฟล์ README.md (optional)
)

print(f"Dataset URL: {dataset_url}")
```

### การใช้เครื่องมือบรรทัดคำสั่ง

สามารถใช้งานฟังก์ชันการอัปโหลดผ่านคำสั่ง `generate_dataset.py` ได้ดังนี้:

```bash
# อัปโหลดไปยัง Hugging Face Hub (ใช้ environment variable HUGGINGFACE_TOKEN)
python src/python/generate_dataset.py sentiment_analysis 100 --export-huggingface --hf-repo-id username/my-dataset

# กำหนด token ผ่าน command line
python src/python/generate_dataset.py sentiment_analysis 100 --export-huggingface --hf-repo-id username/my-dataset --hf-token hf_...

# สร้างเป็น private repository
python src/python/generate_dataset.py sentiment_analysis 100 --export-huggingface --hf-repo-id username/my-dataset --hf-private
```

## การใช้งานชุดข้อมูลจาก Hugging Face Hub

หลังจากอัปโหลดชุดข้อมูลไปยัง Hugging Face Hub แล้ว คุณสามารถใช้งานชุดข้อมูลได้ง่ายดังนี้:

```python
from datasets import load_dataset

# โหลดชุดข้อมูลจาก Hugging Face Hub
dataset = load_dataset("username/my-dataset")

# ดูตัวอย่างข้อมูล
print(dataset["train"][0])

# ใช้งานกับ transformer
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
model = AutoModel.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## ข้อควรระวัง

1. ตรวจสอบว่าข้อมูลของคุณสามารถเผยแพร่ได้อย่างถูกต้องตามกฎหมายและจริยธรรม
2. ระบุ license ที่เหมาะสมสำหรับชุดข้อมูลของคุณ
3. อย่าเผยแพร่ข้อมูลส่วนบุคคลหรือข้อมูลที่ละเอียดอ่อน
4. ระวังขนาดของไฟล์ - ชุดข้อมูลขนาดใหญ่อาจใช้เวลาในการอัปโหลดนาน
