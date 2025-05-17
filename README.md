# Sleep Dataset Generator (Rust)

โปรแกรมสร้างชุดข้อมูลการนอนหลับโดยใช้ DEEPSEEK API สำหรับสร้างข้อมูลที่สมจริง

## ข้อมูลที่สร้าง

1. Sleep Patterns Dataset (8 รูปแบบ)
   - รูปแบบการนอนปกติ, นอนไม่หลับ, ภาวะหยุดหายใจขณะหลับ, ฯลฯ
   - ข้อมูลระยะเวลา คุณภาพ และสัดส่วนของการนอนแต่ละช่วง

2. Sleep Stages Dataset (5 ตัวอย่าง)
   - ข้อมูลการเปลี่ยนแปลงระหว่างช่วงการนอน
   - แสดงลำดับของแต่ละช่วง (awake, light, deep, REM)

3. Sleep Quality Dataset (10 ตัวอย่าง)
   - ข้อมูลคุณภาพการนอนและปัจจัยที่เกี่ยวข้อง
   - สภาพแวดล้อม พฤติกรรม และผลลัพธ์การนอน

## การติดตั้ง

1. ติดตั้ง Rust และ Cargo:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Clone repository:
   ```bash
   git clone <repository-url>
   cd dataset-generator-rs
   ```

3. ตั้งค่า DEEPSEEK API:
   - คัดลอกไฟล์ `.env.example` เป็น `.env`
   - กำหนด DEEPSEEK API key ในไฟล์ `.env`:
     ```
     DEEPSEEK_API_KEY=your_api_key_here
     ```

## การใช้งาน

1. Build โปรแกรม:
   ```bash
   cargo build --release
   ```

2. รันโปรแกรมเพื่อสร้างชุดข้อมูล:
   ```bash
   cargo run
   ```

ชุดข้อมูลที่สร้างขึ้นจะถูกบันทึกในโฟลเดอร์ `data/output/`:
- sleep-patterns-dataset.json
- sleep-stages-dataset.json
- sleep-quality-dataset.json

## โครงสร้างโปรเจค

```
src/
├── main.rs          # Entry point และการจัดการไฟล์
├── models.rs        # Data structures สำหรับชุดข้อมูล
└── api_client.rs    # DEEPSEEK API client
```

## Environment Variables

- `DEEPSEEK_API_KEY`: API key สำหรับเชื่อมต่อกับ DEEPSEEK API (จำเป็นต้องกำหนด)

## การพัฒนาเพิ่มเติม

1. Fork repository
2. สร้าง feature branch
3. Commit changes
4. Push to branch
5. สร้าง Pull Request

## License

MIT