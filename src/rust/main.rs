mod banner;
mod models;
mod generator;
mod task_definitions;
mod api_client;

use dotenv::dotenv;
use models::{DataFormat};
use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use crate::task_definitions::get_task_definitions;
use api_client::DeepseekClient;
use crate::models::GeneratedData;
use std::process::Command;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    banner::print_ascii_banner();
    dotenv().ok();

    // รับ args: task1[,task2,...] count [--parquet|--arrow|--both]
    let args: Vec<String> = env::args().collect();
    let task_names: Vec<&str> = args.get(1)
        .map(|s| s.split(',').collect())
        .unwrap_or_else(|| vec!["sentiment_analysis"]);
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let tasks = get_task_definitions();
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set in environment");
    let client = DeepseekClient::new(api_key);
    let output_dir = Path::new("data/output");
    fs::create_dir_all(output_dir)?;

    // ตรวจสอบ argument ว่าต้องการ Arrow หรือไม่
    let arrow_enabled = args.iter().any(|s| s == "--arrow" || s == "--both");
    let parquet_enabled = !arrow_enabled || args.iter().any(|s| s == "--parquet" || s == "--both");

    for task_name in task_names {
        let task = tasks.get(task_name)
            .unwrap_or_else(|| panic!("Task '{}' not found. Available: {:?}", task_name, tasks.keys()));
        println!("Requesting Deepseek API to generate dataset for task: {} ({} samples)", task.name, count);

        // --- Progress bar ---
        for percent in (0..=100).step_by(10) {
            banner::print_loading_bar(percent);
            std::thread::sleep(std::time::Duration::from_millis(60));
        }
        println!();

        let entries = client.generate_dataset_with_prompt(task, count).await?;
        let data = GeneratedData {
            task_name: task.name.clone(),
            format: task.format.clone(),
            data: entries.iter().enumerate().map(|(i, v)| models::DataEntry {
                id: format!("{}-{}", task_name, i+1),
                content: v.clone(),
                metadata: Some([
                    ("source".to_string(), "DEEPSEEK AI".to_string())
                ].iter().cloned().collect()),
            }).collect(),
        };
        let filename = format!("auto-dataset-{}-{}.{}", task_name, chrono::Local::now().format("%Y%m%d-%H%M%S"), match task.format {
            DataFormat::Parquet => "parquet",
            DataFormat::Arrow => "arrow",
            DataFormat::Json => "json",
            DataFormat::Jsonl => "jsonl",
            DataFormat::Csv => "csv",
            DataFormat::Text => "txt",
            DataFormat::Custom(ref ext) => ext,
            _ => "dat",
        });
        let output_path = output_dir.join(&filename);
        let generator = generator::DatasetGenerator::new();
        let formatted = generator.format_output(&data)?;
        let mut file = File::create(&output_path)?;
        file.write_all(formatted.as_bytes())?;
        println!("Dataset saved to {:?}", output_path);
        // Automate Parquet conversion (always)
        if parquet_enabled {
            let status = Command::new("python")
                .arg("data/output/export_parquet_arrow.py")
                .arg(output_path.to_str().unwrap())
                .arg("parquet")
                .status();
            match status {
                Ok(s) if s.success() => println!("[OK] Converted to parquet for {:?}", output_path),
                Ok(s) => println!("[WARN] Python script exited with status: {s}"),
                Err(e) => println!("[ERR] Could not run Python script: {e}"),
            }
        }
        // Automate Arrow conversion (only if user requested)
        if arrow_enabled {
            let status = Command::new("python")
                .arg("data/output/export_parquet_arrow.py")
                .arg(output_path.to_str().unwrap())
                .arg("arrow")
                .status();
            match status {
                Ok(s) if s.success() => println!("[OK] Converted to arrow for {:?}", output_path),
                Ok(s) => println!("[WARN] Python script exited with status: {s}"),
                Err(e) => println!("[ERR] Could not run Python script: {e}"),
            }
        }
    }

    Ok(())
}