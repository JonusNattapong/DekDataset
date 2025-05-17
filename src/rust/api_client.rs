use crate::models::TaskDefinition;
use reqwest::Client;
use serde_json::Value;

pub struct DeepseekClient {
    client: Client,
    api_key: String,
    api_url: String,
}

impl DeepseekClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            api_url: "https://api.deepseek.com/chat/completions".to_string(),
        }
    }

    pub async fn generate_dataset_with_prompt(&self, task: &TaskDefinition, count: usize) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut results = Vec::new();
        let prompt = format!(
            "คุณคือ AI สำหรับสร้าง dataset โจทย์: {}\nรายละเอียด: {}\nSchema: {:?}\nโปรดสร้างตัวอย่างข้อมูล {} ตัวอย่างในรูปแบบ JSON array ตาม schema ข้างต้น (ไม่ต้องอธิบายเพิ่ม)\n- ทุก field และค่าควรเป็นภาษาไทยล้วน 100% ห้ามมีคำต่างประเทศหรือภาษาจีนในทุก field ยกเว้นกรณีที่เป็นชื่อเฉพาะหรือจำเป็นจริงๆ\n- ข้อมูลต้องสมจริง หลากหลาย ไม่ซ้ำกัน และสอดคล้องกับโจทย์\n- ใช้บริบทและสถานการณ์ที่พบได้จริงในชีวิตประจำวัน\n- หลีกเลี่ยงการใช้ข้อความ placeholder เช่น 'ตัวอย่าง', 'test', 'sample'\n- หาก field เป็น enum หรือ label ให้สุ่มค่าที่เหมาะสมและสมเหตุสมผล\n- หากมี field ที่เป็นตัวเลขหรือวันที่ ให้สุ่มค่าที่สมจริง\n- หลีกเลี่ยงการสร้างข้อมูลที่ขัดแย้งกับตรรกะหรือผิดหลักความเป็นจริง\n- ผลลัพธ์ต้องเป็น JSON array เท่านั้น ไม่ต้องมีคำอธิบายหรือข้อความอื่นประกอบ",
            task.name, task.description, task.schema.fields, count
        );
        let req_body = serde_json::json!({
            "model": "deepseek-chat",
            "temperature": 1.5,
            "messages": [
                {"role": "system", "content": "You are a helpful AI dataset generator."},
                {"role": "user", "content": prompt}
            ]
        });
        let resp = self.client.post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&req_body)
            .send().await?;
        let resp_json: Value = resp.json().await?;
        let mut content = resp_json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
        // Remove code block markers if present
        if content.starts_with("```json") {
            content = content.trim_start_matches("```json").trim().to_string();
        }
        if content.ends_with("```") {
            content = content.trim_end_matches("``` ").trim_end_matches("```").trim().to_string();
        }
        // พยายาม parse เป็น JSON array
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Array(arr) = parsed {
            results.extend(arr);
        }
        Ok(results)
    }
}