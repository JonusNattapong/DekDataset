import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import defaultdict

# --- CONFIG ---
MODEL_DIR = "data/output/thai-education-model"  # หรือใช้ "data/output/thai-education-model-tuned" ถ้ามีโมเดลที่ปรับแล้ว
RESULTS_DIR = "data/output/evaluation-results"
TEST_CASES_DIR = "data/output/test-cases"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_CASES_DIR, exist_ok=True)

# --- โหลดโมเดลและ tokenizer ---
def load_model():
    print(f"โหลดโมเดลจาก {MODEL_DIR}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        subjects = list(model.config.id2label.values())
        print(f"โหลดโมเดลสำเร็จ, รายการวิชา: {subjects}")
        return model, tokenizer, subjects
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลได้: {e}")
        # ลองโหลดจาก WangchanBERTa
        print("ลองโหลดจาก WangchanBERTa แทน...")
        tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", num_labels=8)
        subjects = [
            "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
            "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
        ]
        return model, tokenizer, subjects

# --- สร้างฟังก์ชันทำนาย ---
def predict(text, model, tokenizer, subjects):
    """ทำนายประเภทวิชาของข้อความ"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    prediction_idx = torch.argmax(probabilities).item()
    prediction = subjects[prediction_idx]
    confidence = probabilities[prediction_idx].item()
    
    # ส่งคืนการทำนาย, ความมั่นใจ และความน่าจะเป็นของทุกวิชา
    all_probs = {subjects[i]: prob.item() for i, prob in enumerate(probabilities)}
    return prediction, confidence, all_probs

# --- สร้างตัวอย่างทดสอบตามประเภทวิชา ---
def create_test_cases():
    """สร้างตัวอย่างทดสอบสำหรับแต่ละประเภทวิชา"""
    test_cases = {
        "ภาษาไทย": [
            "ประเภทของคำในภาษาไทย ได้แก่ คำนาม คำสรรพนาม คำกริยา คำวิเศษณ์ คำบุพบท คำสันธาน คำอุทาน",
            "วรรณคดีไทยเรื่อง รามเกียรติ์ เป็นวรรณกรรมที่ได้รับอิทธิพลมาจากมหากาพย์รามายณะของอินเดีย",
            "มาตราตัวสะกดในภาษาไทย ได้แก่ แม่ ก กา แม่กง แม่กม แม่เกย แม่เกอว แม่กก แม่กด แม่กบ",
            "วิธีอ่านบทร้อยกรอง ต้องอ่านตามจังหวะและทำนองให้ถูกต้องตามฉันทลักษณ์",
            "การใช้คำราชาศัพท์ให้ถูกต้องตามกาลเทศะและบุคคล"
        ],
        "คณิตศาสตร์": [
            "การแก้โจทย์ปัญหาสมการกำลังสอง x² - 5x + 6 = 0",
            "ความน่าจะเป็นของการทอดลูกเต๋า 2 ลูกแล้วได้ผลบวกเท่ากับ 7",
            "การหาพื้นที่ใต้กราฟฟังก์ชัน f(x) = x² จาก x=0 ถึง x=2 โดยใช้แคลคูลัส",
            "การคำนวณดอกเบี้ยทบต้น มีสูตร A = P(1 + r/n)^(nt)",
            "ทฤษฎีพีทาโกรัสกล่าวว่า ในรูปสามเหลี่ยมมุมฉาก กำลังสองของด้านตรงข้ามมุมฉากเท่ากับผลบวกของกำลังสองของด้านประกอบมุมฉาก"
        ],
        "วิทยาศาสตร์": [
            "กระบวนการสังเคราะห์แสงของพืช ใช้คลอโรฟิลล์ในการเปลี่ยนคาร์บอนไดออกไซด์และน้ำให้เป็นกลูโคสและออกซิเจน",
            "กฎการเคลื่อนที่ข้อที่สองของนิวตัน: แรงลัพธ์ที่กระทำต่อวัตถุเท่ากับมวลของวัตถุคูณด้วยความเร่ง",
            "โครโมโซมมนุษย์มี 23 คู่ โดย 22 คู่เป็นออโตโซม และ 1 คู่เป็นโครโมโซมเพศ",
            "ตารางธาตุมีการจัดเรียงธาตุตามเลขอะตอมและสมบัติทางเคมี",
            "ความสัมพันธ์ระหว่างสิ่งมีชีวิตในระบบนิเวศ เช่น การพึ่งพาอาศัยกัน การแข่งขัน การล่าเป็นอาหาร"
        ],
        "สังคมศึกษา": [
            "รัฐธรรมนูญแห่งราชอาณาจักรไทยเป็นกฎหมายสูงสุดในการปกครองประเทศ",
            "การปฏิวัติอุตสาหกรรมในยุโรปเริ่มต้นในประเทศอังกฤษราวคริสต์ศตวรรษที่ 18",
            "หลักปรัชญาเศรษฐกิจพอเพียงของพระบาทสมเด็จพระเจ้าอยู่หัว รัชกาลที่ 9",
            "ASEAN หรือสมาคมประชาชาติแห่งเอเชียตะวันออกเฉียงใต้ ก่อตั้งขึ้นเมื่อวันที่ 8 สิงหาคม พ.ศ. 2510",
            "การแบ่งเขตเวลามาตรฐานโลก (Time Zone) โดยใช้เส้นลองจิจูดเป็นเกณฑ์"
        ],
        "ศิลปะ": [
            "องค์ประกอบศิลป์ ได้แก่ เส้น สี รูปร่าง รูปทรง น้ำหนัก ลักษณะผิว พื้นที่ว่าง",
            "ประเภทของดนตรีไทย แบ่งเป็นดนตรีไทยเดิม ดนตรีพื้นบ้าน และดนตรีไทยประยุกต์",
            "ศิลปะการแสดงนาฏศิลป์ไทย เช่น โขน ละคร รำ ระบำ",
            "ความแตกต่างของศิลปะไทยและศิลปะสากล",
            "การผสมสีและวงจรสี ประกอบด้วยสีขั้นที่ 1 คือ แดง เหลือง น้ำเงิน"
        ],
        "สุขศึกษาและพลศึกษา": [
            "การเปลี่ยนแปลงทางร่างกายในวัยรุ่น เช่น การเจริญเติบโตด้านส่วนสูง น้ำหนัก การเปลี่ยนแปลงทางเพศ",
            "การปฐมพยาบาลเบื้องต้นกรณีกระดูกหัก ห้ามเคลื่อนย้ายผู้บาดเจ็บ ให้ใช้ไม้ดามกระดูกก่อน",
            "กติกาของกีฬาฟุตบอล เริ่มเล่นด้วยผู้เล่นฝ่ายละ 11 คน ใช้เวลาแข่งขัน 90 นาที แบ่งเป็น 2 ครึ่ง",
            "หลักการออกกำลังกายที่ถูกต้อง ควรอบอุ่นร่างกายก่อน ออกกำลังกายสม่ำเสมอ และคูลดาวน์หลังออกกำลังกาย",
            "อาหารหลัก 5 หมู่ ที่จำเป็นต่อร่างกาย"
        ],
        "การงานอาชีพและเทคโนโลยี": [
            "การใช้โปรแกรม Microsoft Excel ในการสร้างตารางคำนวณและการใช้สูตรคำนวณพื้นฐาน",
            "วิธีการถนอมอาหาร เช่น การดอง การตากแห้ง การแช่แข็ง การกวน",
            "การจัดการทรัพยากรในการทำงาน ได้แก่ คน เงิน วัสดุอุปกรณ์ และเวลา",
            "การเขียนโปรแกรมคอมพิวเตอร์พื้นฐาน เช่น การใช้คำสั่ง if-else และ loop",
            "การประกอบอาชีพอิสระและการเป็นผู้ประกอบการ"
        ],
        "ภาษาอังกฤษ": [
            "The Present Perfect Tense is used to express an action that happened in the past but has a connection to the present.",
            "Modal verbs in English include can, could, may, might, shall, should, will, would, must.",
            "Passive Voice is used when the focus is on the action rather than who is performing the action.",
            "Conditional sentences type 1: 'If it rains tomorrow, I will stay at home.'",
            "Using articles correctly: 'a', 'an', and 'the' in English grammar."
        ]
    }
    
    # บันทึกตัวอย่างทดสอบลงไฟล์
    test_cases_file = os.path.join(TEST_CASES_DIR, "subject_test_cases.json")
    with open(test_cases_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"บันทึกตัวอย่างทดสอบไปยัง {test_cases_file}")
    
    return test_cases

# --- ประเมินโมเดลด้วยตัวอย่างทดสอบ ---
def evaluate_model(model, tokenizer, subjects, test_cases):
    """ประเมินโมเดลด้วยตัวอย่างทดสอบของแต่ละวิชา"""
    results = []
    subject_true = []
    subject_pred = []
    
    for subject, texts in test_cases.items():
        print(f"\nทดสอบวิชา: {subject}")
        for text in texts:
            prediction, confidence, all_probs = predict(text, model, tokenizer, subjects)
            print(f"ข้อความ: {text[:50]}... => ทำนาย: {prediction} ({confidence:.2%})")
            
            # เก็บผลลัพธ์
            results.append({
                "subject_true": subject,
                "subject_pred": prediction,
                "text": text,
                "confidence": confidence,
                "all_probabilities": all_probs
            })
            
            # เก็บข้อมูลสำหรับคำนวณ metrics
            subject_true.append(subject)
            subject_pred.append(prediction)
    
    # คำนวณ metrics
    accuracy = accuracy_score(subject_true, subject_pred)
    f1 = f1_score(subject_true, subject_pred, average="macro")
    conf_matrix = confusion_matrix(subject_true, subject_pred, labels=subjects)
    
    # สร้างรายงาน
    print("\n====== ผลการประเมินโมเดล ======")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(subject_true, subject_pred, labels=subjects, target_names=subjects))
    
    # บันทึกผลลัพธ์
    results_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "f1_score": f1,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"บันทึกผลการประเมินไปยัง {results_file}")
    
    # สร้างและบันทึก confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=subjects, yticklabels=subjects)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"บันทึก Confusion Matrix ไปยัง {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")
    
    # วิเคราะห์จุดอ่อน (ตัวอย่างที่ทำนายผิด)
    incorrect_predictions = [res for res in results if res["subject_true"] != res["subject_pred"]]
    if incorrect_predictions:
        print("\n====== ตัวอย่างที่ทำนายผิด ======")
        error_analysis = defaultdict(list)
        for res in incorrect_predictions:
            error_analysis[(res["subject_true"], res["subject_pred"])].append({
                "text": res["text"],
                "confidence": res["confidence"],
                "all_probs": res["all_probabilities"]
            })
        
        # บันทึกการวิเคราะห์ข้อผิดพลาด
        error_file = os.path.join(RESULTS_DIR, "error_analysis.json")
        with open(error_file, "w", encoding="utf-8") as f:
            error_dict = {f"{true}->{pred}": data for (true, pred), data in error_analysis.items()}
            json.dump(error_dict, f, ensure_ascii=False, indent=2)
        print(f"บันทึกการวิเคราะห์ข้อผิดพลาดไปยัง {error_file}")
        
        # แสดงตัวอย่างข้อผิดพลาดบางส่วน
        for (true, pred), examples in list(error_analysis.items())[:3]:
            print(f"\nวิชาจริง: {true} -> ทำนายเป็น: {pred}")
            for i, ex in enumerate(examples[:2]):
                print(f"  {i+1}. '{ex['text'][:50]}...' (ความเชื่อมั่น: {ex['confidence']:.2%})")
    
    return accuracy, f1, results

# --- ฟังก์ชันหลัก ---
def main():
    # 1. โหลดโมเดล
    model, tokenizer, subjects = load_model()
    
    # 2. สร้างตัวอย่างทดสอบ
    test_cases = create_test_cases()
    
    # 3. ทดสอบโมเดล
    accuracy, f1, results = evaluate_model(model, tokenizer, subjects, test_cases)
    
    # 4. ทดสอบกับตัวอย่างเฉพาะที่น่าสนใจ
    print("\n====== ทดสอบกับตัวอย่างพิเศษ ======")
    special_cases = [
        "การวิเคราะห์บทกวีนิพนธ์และคำประพันธ์ประเภทฉันท์ กาพย์ กลอน",  # ภาษาไทยที่ซับซ้อน
        "การแก้ปัญหาเรื่อง pH ของสารละลาย H₂SO₄ เข้มข้น 0.1 โมลาร์",  # วิทยาศาสตร์ผสมคณิตศาสตร์
        "การหาค่า GDP และอัตราเงินเฟ้อในระบบเศรษฐกิจ",  # สังคมศึกษาผสมคณิตศาสตร์
        "HTML และ CSS เป็นภาษาพื้นฐานในการพัฒนาเว็บไซต์",  # การงานอาชีพฯ
        "Football is played in the World Cup tournament every four years.",  # ภาษาอังกฤษผสมพลศึกษา
        "มนุษย์และสิ่งแวดล้อมอยู่ร่วมกันอย่างสมดุล"  # คำกลางๆ ที่อาจจะเข้าได้หลายวิชา
    ]
    
    for text in special_cases:
        prediction, confidence, all_probs = predict(text, model, tokenizer, subjects)
        print(f"ข้อความ: {text}")
        print(f"ทำนาย: {prediction} (ความมั่นใจ: {confidence:.2%})")
        
        # แสดงวิชาทั้งหมดเรียงตามความน่าจะเป็น
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        print("ความน่าจะเป็นของแต่ละวิชา:")
        for subj, prob in sorted_probs:
            print(f"  {subj}: {prob:.2%}")
        print()
    
    print(f"\nการประเมินเสร็จสิ้น! ความแม่นยำโดยรวม: {accuracy:.2%}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
