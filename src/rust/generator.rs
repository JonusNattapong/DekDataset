use crate::models::{GeneratedData, DataFormat};
use csv;

pub struct DatasetGenerator;

impl DatasetGenerator {
    pub fn new() -> Self {
        DatasetGenerator
    }
    pub fn format_output(&self, data: &GeneratedData) -> anyhow::Result<String> {
        match data.format {
            DataFormat::Json => {
                Ok(serde_json::to_string_pretty(&data.data)?)
            }
            DataFormat::Jsonl => {
                let lines: anyhow::Result<Vec<String>> = data.data.iter()
                    .map(|entry| Ok(serde_json::to_string(entry)?))
                    .collect();
                Ok(lines?.join("\n"))
            }
            DataFormat::Csv => {
                let mut wtr = csv::Writer::from_writer(vec![]);
                let mut header_written = false;
                for entry in &data.data {
                    if let Some(obj) = entry.content.as_object() {
                        if !header_written {
                            let headers: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                            wtr.write_record(&headers)?;
                            header_written = true;
                        }
                        let row: Vec<String> = obj.values().map(|v| v.to_string()).collect();
                        wtr.write_record(&row)?;
                    }
                }
                let data = wtr.into_inner()?;
                Ok(String::from_utf8(data)?)
            }
            DataFormat::Parquet => {
                Ok("Parquet export is not implemented in Rust. Please use Python/CLI for this format.".to_string())
            }
            DataFormat::Arrow => {
                Ok("Arrow export is not implemented in Rust. Please use Python/CLI for this format.".to_string())
            }
            DataFormat::Text => {
                let lines: Vec<String> = data.data.iter()
                    .filter_map(|entry| entry.content.get("text").and_then(|s| s.as_str().map(|s| s.to_string())))
                    .collect();
                Ok(lines.join("\n"))
            }
            DataFormat::WebDataset | DataFormat::Custom(_) => {
                Err(anyhow::anyhow!("This format is not supported for direct export."))
            }
        }
    }
}