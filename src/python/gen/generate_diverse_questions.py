#!/usr/bin/env python
# coding: utf-8
# generate_diverse_questions.py - สร้างข้อคำถามที่หลากหลายสำหรับแต่ละวิชา

import os
import json
import random
from tqdm import tqdm
import argparse

# คำถามตัวอย่างสำหรับแต่ละวิชา
SUBJECT_TEMPLATES = {
    "ภาษาไทย": [
        {"question": "คำว่า '{word}' มีความหมายตรงกับข้อใด", "choices": ["ความหมาย 1", "ความหมาย 2", "ความหมาย 3", "ความหมาย 4"]},
        {"question": "ข้อใดเป็นคำ{word_type}", "choices": ["คำตัวอย่าง 1", "คำตัวอย่าง 2", "คำตัวอย่าง 3", "คำตัวอย่าง 4"]}, 
        {"question": "คำในข้อใดมีพยางค์มากที่สุด", "choices": ["คำตัวอย่าง 1", "คำตัวอย่าง 2", "คำตัวอย่าง 3", "คำตัวอย่าง 4"]},
        {"question": "คำในข้อใดเป็นคำที่มีความหมายโดยนัย", "choices": ["คำตัวอย่าง 1", "คำตัวอย่าง 2", "คำตัวอย่าง 3", "คำตัวอย่าง 4"]},
        {"question": "ข้อใดเป็นสำนวนไทย", "choices": ["สำนวน 1", "สำนวน 2", "สำนวน 3", "สำนวน 4"]},
    ],
    "คณิตศาสตร์": [
        {"question": "ข้อใดเป็นคำตอบของ {number1} + {number2}", "choices": ["{answer}", "{wrong1}", "{wrong2}", "{wrong3}"]},
        {"question": "{number1} - {number2} เท่ากับเท่าไร", "choices": ["{answer}", "{wrong1}", "{wrong2}", "{wrong3}"]},
        {"question": "ข้อใดเป็นผลลัพธ์ของ {number1} × {number2}", "choices": ["{answer}", "{wrong1}", "{wrong2}", "{wrong3}"]},
        {"question": "{number1} ÷ {number2} เท่ากับข้อใด", "choices": ["{answer}", "{wrong1}", "{wrong2}", "{wrong3}"]},
        {"question": "ข้อใดเป็นค่าของ x ในสมการ {equation}", "choices": ["{answer}", "{wrong1}", "{wrong2}", "{wrong3}"]},
    ],
    "วิทยาศาสตร์": [
        {"question": "ข้อใดเป็น{science_topic}", "choices": ["คำตอบ 1", "คำตอบ 2", "คำตอบ 3", "คำตอบ 4"]},
        {"question": "ข้อใดไม่ใช่{science_type}", "choices": ["คำตอบ 1", "คำตอบ 2", "คำตอบ 3", "คำตอบ 4"]},
        {"question": "{process} เกี่ยวข้องกับข้อใดมากที่สุด", "choices": ["คำตอบ 1", "คำตอบ 2", "คำตอบ 3", "คำตอบ 4"]},
        {"question": "ข้อใดไม่ใช่ลักษณะของ{organism}", "choices": ["คำตอบ 1", "คำตอบ 2", "คำตอบ 3", "คำตอบ 4"]},
        {"question": "ข้อใดเป็นประโยชน์ของ{element}", "choices": ["คำตอบ 1", "คำตอบ 2", "คำตอบ 3", "คำตอบ 4"]},
    ],
    "สังคมศึกษา": [
        {"question": "ข้อใดเป็นเหตุการณ์ที่เกิดขึ้นในรัชสมัย{king}", "choices": ["เหตุการณ์ 1", "เหตุการณ์ 2", "เหตุการณ์ 3", "เหตุการณ์ 4"]},
        {"question": "ข้อใดไม่ใช่ลักษณะของ{geography_feature}", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "วัฒนธรรม{culture_type}มีลักษณะตรงกับข้อใด", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "ข้อใดเป็นหลักการสำคัญของศาสนา{religion}", "choices": ["หลักการ 1", "หลักการ 2", "หลักการ 3", "หลักการ 4"]},
        {"question": "ข้อใดเป็นผลกระทบของ{event}ต่อสังคมไทย", "choices": ["ผลกระทบ 1", "ผลกระทบ 2", "ผลกระทบ 3", "ผลกระทบ 4"]},
    ],
    "ภาษาอังกฤษ": [
        {"question": "What is the meaning of '{word}'?", "choices": ["meaning 1", "meaning 2", "meaning 3", "meaning 4"]},
        {"question": "Which sentence is correct?", "choices": ["sentence 1", "sentence 2", "sentence 3", "sentence 4"]},
        {"question": "Choose the correct tense for this sentence: {sentence}", "choices": ["tense 1", "tense 2", "tense 3", "tense 4"]},
        {"question": "What is the opposite of '{word}'?", "choices": ["opposite 1", "opposite 2", "opposite 3", "opposite 4"]},
        {"question": "Which word has the same meaning as '{word}'?", "choices": ["synonym 1", "synonym 2", "synonym 3", "synonym 4"]},
    ],
    "ศิลปะ": [
        {"question": "ข้อใดเป็นองค์ประกอบของ{art_form}", "choices": ["องค์ประกอบ 1", "องค์ประกอบ 2", "องค์ประกอบ 3", "องค์ประกอบ 4"]},
        {"question": "ศิลปิน{artist}มีผลงานที่โดดเด่นตรงกับข้อใด", "choices": ["ผลงาน 1", "ผลงาน 2", "ผลงาน 3", "ผลงาน 4"]},
        {"question": "ข้อใดไม่ใช่ลักษณะของศิลปะ{art_style}", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "เทคนิค{technique}ในงานศิลปะมีลักษณะอย่างไร", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "องค์ประกอบศิลป์ข้อใดสำคัญที่สุดในงาน{art_work}", "choices": ["องค์ประกอบ 1", "องค์ประกอบ 2", "องค์ประกอบ 3", "องค์ประกอบ 4"]},
    ],
    "สุขศึกษาและพลศึกษา": [
        {"question": "ข้อใดเป็นประโยชน์ของการ{exercise}", "choices": ["ประโยชน์ 1", "ประโยชน์ 2", "ประโยชน์ 3", "ประโยชน์ 4"]},
        {"question": "ข้อใดเป็นวิธีป้องกัน{disease}", "choices": ["วิธี 1", "วิธี 2", "วิธี 3", "วิธี 4"]},
        {"question": "กีฬา{sport}มีกติกาข้อใดที่สำคัญที่สุด", "choices": ["กติกา 1", "กติกา 2", "กติกา 3", "กติกา 4"]},
        {"question": "ข้อใดไม่ใช่อาการของ{health_condition}", "choices": ["อาการ 1", "อาการ 2", "อาการ 3", "อาการ 4"]},
        {"question": "การดูแลสุขภาพ{body_part}ควรทำอย่างไร", "choices": ["วิธี 1", "วิธี 2", "วิธี 3", "วิธี 4"]},
    ],
    "การงานอาชีพและเทคโนโลยี": [
        {"question": "ข้อใดเป็นขั้นตอนที่ถูกต้องในการ{task}", "choices": ["ขั้นตอน 1", "ขั้นตอน 2", "ขั้นตอน 3", "ขั้นตอน 4"]},
        {"question": "ข้อใดไม่ใช่ประโยชน์ของ{technology}", "choices": ["ประโยชน์ 1", "ประโยชน์ 2", "ประโยชน์ 3", "ประโยชน์ 4"]},
        {"question": "เครื่องมือ{tool}ใช้สำหรับอะไร", "choices": ["การใช้งาน 1", "การใช้งาน 2", "การใช้งาน 3", "การใช้งาน 4"]},
        {"question": "ข้อใดเป็นวิธีการ{method}ที่ถูกต้อง", "choices": ["วิธีการ 1", "วิธีการ 2", "วิธีการ 3", "วิธีการ 4"]},
        {"question": "อาชีพ{occupation}มีลักษณะงานอย่างไร", "choices": ["ลักษณะงาน 1", "ลักษณะงาน 2", "ลักษณะงาน 3", "ลักษณะงาน 4"]},
    ],
    "ดนตรี": [
        {"question": "เครื่องดนตรี{instrument}จัดอยู่ในประเภทใด", "choices": ["ประเภท 1", "ประเภท 2", "ประเภท 3", "ประเภท 4"]},
        {"question": "ข้อใดไม่ใช่ลักษณะของดนตรี{music_style}", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "ข้อใดเป็นจังหวะดนตรีแบบ{rhythm}", "choices": ["จังหวะ 1", "จังหวะ 2", "จังหวะ 3", "จังหวะ 4"]},
        {"question": "ดนตรี{music_type}มีต้นกำเนิดจากที่ใด", "choices": ["แหล่ง 1", "แหล่ง 2", "แหล่ง 3", "แหล่ง 4"]},
        {"question": "ศิลปิน{musician}มีผลงานเพลงตรงกับข้อใด", "choices": ["เพลง 1", "เพลง 2", "เพลง 3", "เพลง 4"]},
    ],
    "เทคโนโลยี": [
        {"question": "เทคโนโลยี{tech_name}มีประโยชน์ตรงกับข้อใด", "choices": ["ประโยชน์ 1", "ประโยชน์ 2", "ประโยชน์ 3", "ประโยชน์ 4"]},
        {"question": "ข้อใดไม่ใช่องค์ประกอบของ{system}", "choices": ["องค์ประกอบ 1", "องค์ประกอบ 2", "องค์ประกอบ 3", "องค์ประกอบ 4"]},
        {"question": "อุปกรณ์{device}ทำงานอย่างไร", "choices": ["การทำงาน 1", "การทำงาน 2", "การทำงาน 3", "การทำงาน 4"]},
        {"question": "การพัฒนา{software}มีขั้นตอนอย่างไร", "choices": ["ขั้นตอน 1", "ขั้นตอน 2", "ขั้นตอน 3", "ขั้นตอน 4"]},
        {"question": "ข้อใดเป็นผลกระทบของ{tech_impact}", "choices": ["ผลกระทบ 1", "ผลกระทบ 2", "ผลกระทบ 3", "ผลกระทบ 4"]},
    ],
    "ภาษาจีน": [
        {"question": "คำว่า '{chinese_word}' มีความหมายตรงกับข้อใด", "choices": ["ความหมาย 1", "ความหมาย 2", "ความหมาย 3", "ความหมาย 4"]},
        {"question": "ข้อใดเป็นประโยค{sentence_type}ในภาษาจีน", "choices": ["ประโยค 1", "ประโยค 2", "ประโยค 3", "ประโยค 4"]},
        {"question": "คำในข้อใดออกเสียงเป็น {pinyin}", "choices": ["คำ 1", "คำ 2", "คำ 3", "คำ 4"]},
        {"question": "ข้อใดเป็นสำนวนจีนที่มีความหมายว่า {idiom_meaning}", "choices": ["สำนวน 1", "สำนวน 2", "สำนวน 3", "สำนวน 4"]},
        {"question": "วัฒนธรรมจีนเกี่ยวกับ{culture_topic}มีลักษณะอย่างไร", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
    ],
    "ประวัติศาสตร์": [
        {"question": "เหตุการณ์{event}เกิดขึ้นในสมัยใด", "choices": ["สมัย 1", "สมัย 2", "สมัย 3", "สมัย 4"]},
        {"question": "บุคคลสำคัญ{person}มีบทบาทอย่างไรในประวัติศาสตร์", "choices": ["บทบาท 1", "บทบาท 2", "บทบาท 3", "บทบาท 4"]},
        {"question": "สงคราม{war}มีสาเหตุมาจากข้อใด", "choices": ["สาเหตุ 1", "สาเหตุ 2", "สาเหตุ 3", "สาเหตุ 4"]},
        {"question": "ข้อใดเป็นผลกระทบของ{historical_event}", "choices": ["ผลกระทบ 1", "ผลกระทบ 2", "ผลกระทบ 3", "ผลกระทบ 4"]},
        {"question": "อาณาจักร{kingdom}มีอาณาเขตตรงกับข้อใด", "choices": ["อาณาเขต 1", "อาณาเขต 2", "อาณาเขต 3", "อาณาเขต 4"]},
    ],
    "ภูมิศาสตร์": [
        {"question": "ข้อใดเป็นลักษณะภูมิประเทศของ{location}", "choices": ["ลักษณะ 1", "ลักษณะ 2", "ลักษณะ 3", "ลักษณะ 4"]},
        {"question": "ภูมิอากาศแบบ{climate_type}พบได้ในบริเวณใด", "choices": ["บริเวณ 1", "บริเวณ 2", "บริเวณ 3", "บริเวณ 4"]},
        {"question": "แม่น้ำ{river}ไหลผ่านจังหวัดใดบ้าง", "choices": ["จังหวัด 1", "จังหวัด 2", "จังหวัด 3", "จังหวัด 4"]},
        {"question": "ข้อใดไม่ใช่ทรัพยากรธรรมชาติที่พบในภาค{region}", "choices": ["ทรัพยากร 1", "ทรัพยากร 2", "ทรัพยากร 3", "ทรัพยากร 4"]},
        {"question": "ปรากฏการณ์{phenomenon}เกิดจากสาเหตุใด", "choices": ["สาเหตุ 1", "สาเหตุ 2", "สาเหตุ 3", "สาเหตุ 4"]},
    ]
}

# ตัวแปรแทนที่ใช้ใน template
PLACEHOLDERS = {
    "word": ["ราชาศัพท์", "พยัญชนะ", "สระ", "วรรณยุกต์", "อักษรนำ", "คำราชาศัพท์", "คำสมาส", "คำสนธิ", "คำทับศัพท์", "คำซ้อน"],
    "word_type": ["นาม", "สรรพนาม", "กริยา", "วิเศษณ์", "บุพบท", "สันธาน", "อุทาน"],
    "number1": [str(i) for i in range(1, 100)],
    "number2": [str(i) for i in range(1, 100)],
    "equation": ["2x + 3 = 11", "5x - 2 = 13", "3x + 5 = 17", "4x - 7 = 9", "2x - 5 = 15"],
    "science_topic": ["วงจรชีวิตของผีเสื้อ", "การเคลื่อนที่ของโลก", "ระบบย่อยอาหาร", "การเปลี่ยนสถานะของสสาร", "การถ่ายทอดพลังงาน"],
    "science_type": ["สัตว์เลี้ยงลูกด้วยนม", "สัตว์เลื้อยคลาน", "สัตว์ครึ่งบกครึ่งน้ำ", "โลหะ", "อโลหะ"],
    "process": ["การสังเคราะห์แสง", "การหายใจ", "การแบ่งเซลล์", "การถ่ายทอดลักษณะทางพันธุกรรม", "การหมุนเวียนของน้ำ"],
    "organism": ["สัตว์เลี้ยงลูกด้วยนม", "ปลา", "นก", "แมลง", "พืชดอก"],
    "element": ["ออกซิเจน", "คาร์บอน", "ไนโตรเจน", "ไฮโดรเจน", "เหล็ก"],
    "king": ["รัชกาลที่ 5", "รัชกาลที่ 9", "พ่อขุนรามคำแหงมหาราช", "สมเด็จพระนารายณ์มหาราช", "สมเด็จพระนเรศวรมหาราช"],
    "geography_feature": ["ภูเขา", "แม่น้ำ", "ทะเล", "ทะเลสาบ", "เกาะ"],
    "culture_type": ["ล้านนา", "อีสาน", "ภาคใต้", "ภาคกลาง", "ไทย"],
    "religion": ["พุทธ", "คริสต์", "อิสลาม", "ฮินดู", "ซิกข์"],
    "event": ["การปฏิวัติ 2475", "สนธิสัญญาเบาว์ริง", "การพัฒนาเศรษฐกิจและสังคมแห่งชาติ", "วิกฤติเศรษฐกิจ 2540", "การเข้าร่วมสงครามโลกครั้งที่ 1"],
    "art_form": ["จิตรกรรม", "ประติมากรรม", "สถาปัตยกรรม", "ศิลปะการแสดง", "หัตถกรรม"],
    "artist": ["เฉลิมชัย โฆษิตพิพัฒน์", "ถวัลย์ ดัชนี", "อังคาร กัลยาณพงศ์", "ศิลป์ พีระศรี", "ทวี รัชนีกร"],
    "art_style": ["ไทยประเพณี", "จิตรกรรมฝาผนัง", "ศิลปะร่วมสมัย", "อิมเพรสชันนิสม์", "เอกซ์เพรสชันนิสม์"],
    "technique": ["สีน้ำ", "สีน้ำมัน", "ปูนปั้น", "แกะสลัก", "ภาพพิมพ์"],
    "art_work": ["ภาพจิตรกรรม", "ประติมากรรม", "สถาปัตยกรรม", "ภาพพิมพ์", "ภาพถ่าย"],
    "exercise": ["วิ่ง", "ว่ายน้ำ", "เดิน", "ปั่นจักรยาน", "โยคะ"],
    "disease": ["ไข้หวัด", "โรคเบาหวาน", "โรคความดันโลหิตสูง", "โรคหัวใจ", "โรคมะเร็ง"],
    "sport": ["ฟุตบอล", "บาสเกตบอล", "วอลเลย์บอล", "เทนนิส", "ว่ายน้ำ"],
    "health_condition": ["โรคหวัด", "โรคภูมิแพ้", "โรคกระเพาะ", "โรคไมเกรน", "โรคผิวหนัง"],
    "body_part": ["ฟัน", "ผิวหนัง", "ดวงตา", "เส้นผม", "กล้ามเนื้อ"],
    "task": ["ทำอาหาร", "ปลูกต้นไม้", "ซ่อมแซมเครื่องใช้ไฟฟ้า", "เย็บผ้า", "ทำความสะอาดบ้าน"],
    "technology": ["อินเทอร์เน็ต", "สมาร์ทโฟน", "คอมพิวเตอร์", "หุ่นยนต์", "พลังงานทดแทน"],
    "tool": ["ค้อน", "ไขควง", "เลื่อย", "คีม", "สว่าน"],
    "method": ["เพาะปลูกพืช", "ถนอมอาหาร", "ทำความสะอาด", "จัดเก็บข้อมูล", "ประหยัดพลังงาน"],
    "occupation": ["ครู", "แพทย์", "วิศวกร", "นักบัญชี", "เกษตรกร"],
    "instrument": ["ขิม", "ระนาด", "ซอ", "กลอง", "ปี่"],
    "music_style": ["ไทยเดิม", "ลูกทุ่ง", "ลูกกรุง", "สากล", "คลาสสิก"],
    "rhythm": ["จังหวะช้า", "จังหวะเร็ว", "จังหวะสามชั้น", "จังหวะสองชั้น", "จังหวะชั้นเดียว"],
    "music_type": ["ดนตรีไทย", "ดนตรีลูกทุ่ง", "ดนตรีคลาสสิก", "ดนตรีแจ๊ส", "ดนตรีร็อค"],
    "musician": ["คาราบาว", "สุนทราภรณ์", "พงษ์เทพ กระโดนชำนาญ", "ธงไชย แมคอินไตย์", "เบิร์ด ธงไชย"],
    "tech_name": ["ปัญญาประดิษฐ์", "อินเทอร์เน็ตของสรรพสิ่ง", "เทคโนโลยีเสมือนจริง", "บล็อกเชน", "การพิมพ์สามมิติ"],
    "system": ["คอมพิวเตอร์", "เครือข่าย", "ฐานข้อมูล", "ระบบปฏิบัติการ", "ระบบนิเวศดิจิทัล"],
    "device": ["สมาร์ทโฟน", "คอมพิวเตอร์", "แท็บเล็ต", "สมาร์ทวอทช์", "อุปกรณ์อัจฉริยะ"],
    "software": ["แอปพลิเคชัน", "เว็บไซต์", "ระบบปฏิบัติการ", "เกม", "โปรแกรมประมวลผลคำ"],
    "tech_impact": ["สื่อสังคมออนไลน์", "การทำงานทางไกล", "การเรียนออนไลน์", "การซื้อขายออนไลน์", "เทคโนโลยีด้านสุขภาพ"],
    "chinese_word": ["你好", "谢谢", "学生", "老师", "中国"],
    "sentence_type": ["ทักทาย", "แนะนำตัว", "บอกเล่า", "คำถาม", "ปฏิเสธ"],
    "pinyin": ["nǐ hǎo", "xiè xiè", "xué shēng", "lǎo shī", "zhōng guó"],
    "idiom_meaning": ["อดทน", "ความขยัน", "ความเพียร", "มิตรภาพ", "ความกตัญญู"],
    "culture_topic": ["เทศกาลตรุษจีน", "การเขียนพู่กันจีน", "การปรุงอาหารจีน", "การแพทย์แผนจีน", "ศิลปะการต่อสู้แบบจีน"],
    "event": ["กรุงศรีอยุธยาแตก", "การปฏิวัติ 2475", "การเข้าร่วมสงครามโลกครั้งที่ 2", "การสถาปนากรุงรัตนโกสินทร์", "การทำสนธิสัญญาเบาว์ริง"],
    "person": ["สมเด็จพระนเรศวรมหาราช", "รัชกาลที่ 5", "ปรีดี พนมยงค์", "จอมพล ป. พิบูลสงคราม", "พระยาพหลพลพยุหเสนา"],
    "war": ["สงครามเก้าทัพ", "สงครามยุทธหัตถี", "สงครามโลกครั้งที่ 2", "สงครามเย็น", "สงครามมหาเอเชียบูรพา"],
    "historical_event": ["การปฏิวัติอุตสาหกรรม", "การค้นพบทวีปอเมริกา", "การปฏิวัติฝรั่งเศส", "สงครามโลกครั้งที่ 1", "การปฏิวัติรัสเซีย"],
    "kingdom": ["สุโขทัย", "อยุธยา", "ธนบุรี", "รัตนโกสินทร์", "ล้านนา"],
    "location": ["ภาคเหนือ", "ภาคใต้", "ภาคกลาง", "ภาคตะวันออกเฉียงเหนือ", "ภาคตะวันออก"],
    "climate_type": ["ร้อนชื้น", "อบอุ่น", "หนาว", "แห้งแล้ง", "มรสุม"],
    "river": ["เจ้าพระยา", "ป่าสัก", "แม่กลอง", "ปิง", "วัง"],
    "region": ["เหนือ", "กลาง", "ตะวันออกเฉียงเหนือ", "ตะวันออก", "ใต้"],
    "phenomenon": ["ฝนตก", "แผ่นดินไหว", "การกัดเซาะชายฝั่ง", "การเกิดภูเขาไฟ", "สึนามิ"],
}

def fill_template(template, grade):
    """แทนที่ placeholder ในแม่แบบด้วยข้อมูลจริง"""
    result = template.copy()
    
    # แทนที่ placeholder ในคำถาม
    question = result["question"]
    for placeholder in PLACEHOLDERS:
        if "{" + placeholder + "}" in question:
            # สุ่มเลือกข้อมูลจาก placeholder
            value = random.choice(PLACEHOLDERS[placeholder])
            question = question.replace("{" + placeholder + "}", value)
            
            # หากเป็นข้อมูลตัวเลข ให้คำนวณเพื่อใช้ในคำตอบ
            if placeholder == "number1" and "{answer}" in str(result["choices"]):
                number1 = int(value)
                # หาคำตอบและตัวเลือกผิด
                if "{number2}" in question:
                    number2_str = random.choice(PLACEHOLDERS["number2"])
                    number2 = int(number2_str)
                    question = question.replace("{number2}", number2_str)
                    
                    if "+" in question:
                        answer = number1 + number2
                    elif "-" in question:
                        answer = number1 - number2
                    elif "×" in question or "x" in question.lower():
                        answer = number1 * number2
                    elif "÷" in question or "/" in question:
                        # ป้องกันการหารด้วยศูนย์
                        if number2 == 0:
                            number2 = 1
                            question = question.replace("÷ 0", "÷ 1")
                            question = question.replace("/ 0", "/ 1")
                        answer = number1 // number2  # ใช้การหารเอาส่วน
                    
                    # สร้างตัวเลือกที่ผิด
                    wrong_answers = [answer + random.randint(1, 5) for _ in range(3)]
                    choices = [str(answer)] + [str(w) for w in wrong_answers]
                    random.shuffle(choices)
                    
                    # แทนที่ตัวเลือก
                    answer_index = choices.index(str(answer))
                    for i in range(len(result["choices"])):
                        choice_str = result["choices"][i]
                        if "{answer}" in choice_str and i == answer_index:
                            result["choices"][i] = str(answer)
                        elif "{wrong1}" in choice_str:
                            result["choices"][i] = str(wrong_answers[0])
                        elif "{wrong2}" in choice_str:
                            result["choices"][i] = str(wrong_answers[1])
                        elif "{wrong3}" in choice_str:
                            result["choices"][i] = str(wrong_answers[2])
    
    # ตั้งค่าคำถาม
    result["question"] = question
    
    # ตั้งค่าระดับชั้น
    result["grade"] = grade
    
    # สุ่มคำตอบ (หากไม่ได้กำหนดไว้)
    if all("{" not in choice for choice in result["choices"]):
        result["answer"] = random.choice(result["choices"])
    else:
        # กรณีที่ตัวเลือกยังมี placeholder ให้แทนที่ด้วยข้อความอื่น
        for i in range(len(result["choices"])):
            if "{" in result["choices"][i]:
                result["choices"][i] = f"ตัวเลือกที่ {i+1}"
        result["answer"] = random.choice(result["choices"])
    
    # เพิ่มระดับความยาก
    result["difficulty"] = random.choice(["ง่าย", "กลาง", "ยาก"])
    
    # เพิ่ม tag
    result["tags"] = ["dataset_generated", f"grade_{grade}", result["subject"].lower()]
    
    # เพิ่มปีข้อมูล (ปัจจุบัน)
    result["year"] = 2025
    
    # เพิ่มแหล่งที่มา
    result["source"] = "auto-generated"
    
    return result

def generate_questions(subjects, grades, questions_per_grade=10):
    """สร้างคำถามสำหรับแต่ละวิชาและระดับชั้น"""
    generated_data = []
    
    for subject in tqdm(subjects, desc="Generating questions by subject"):
        if subject not in SUBJECT_TEMPLATES:
            print(f"ไม่พบแม่แบบสำหรับวิชา {subject} ข้ามไป")
            continue
        
        for grade in grades:
            templates = SUBJECT_TEMPLATES[subject]
            
            for _ in range(questions_per_grade):
                # สุ่มเลือกแม่แบบสำหรับวิชานี้
                template = random.choice(templates)
                
                # เพิ่มข้อมูลวิชา
                template_copy = template.copy()
                template_copy["subject"] = subject
                
                # แทนที่ placeholder และสร้างคำถาม
                question = fill_template(template_copy, grade)
                generated_data.append(question)
    
    return generated_data

def main():
    parser = argparse.ArgumentParser(description="สร้างข้อคำถามที่หลากหลายสำหรับแต่ละวิชา")
    parser.add_argument("--output", required=True, help="ไฟล์ JSONL ที่จะบันทึกข้อมูลที่สร้าง")
    parser.add_argument("--questions_per_grade", type=int, default=10, help="จำนวนคำถามต่อระดับชั้นต่อวิชา")
    args = parser.parse_args()
    
    # กำหนดรายวิชาและระดับชั้น
    subjects = list(SUBJECT_TEMPLATES.keys())
    primary_grades = list(range(1, 7))  # ป.1-6
    secondary_grades = list(range(7, 13))  # ม.1-6
    all_grades = primary_grades + secondary_grades
    
    # สร้างคำถาม
    generated_questions = generate_questions(subjects, all_grades, args.questions_per_grade)
    
    # บันทึกข้อมูล
    with open(args.output, "w", encoding="utf-8") as f:
        for item in generated_questions:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"สร้างข้อคำถามทั้งหมด {len(generated_questions)} ข้อ และบันทึกไปยัง {args.output}")

if __name__ == "__main__":
    main()
