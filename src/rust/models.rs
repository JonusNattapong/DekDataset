use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// --- DataFormat, Domain, Context, Template, Annotation, Scenario, Example (เดิม) ---
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DataFormat {
    Json,
    Jsonl,
    Csv,
    Parquet,
    WebDataset,
    Text,
    Arrow,
    Custom(String),
}

// --- สำหรับ Task Definition/Schema/Generator ---
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskDefinition {
    pub name: String,
    pub description: String,
    pub format: DataFormat,
    pub schema: DataSchema,
    pub examples: Vec<String>,
    pub parameters: HashMap<String, Parameter>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DataSchema {
    pub fields: HashMap<String, FieldDefinition>,
    pub relationships: Option<Vec<Relationship>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FieldDefinition {
    pub field_type: FieldType,
    pub required: bool,
    pub description: String,
    pub constraints: Option<Vec<Constraint>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum FieldType {
    Text,
    Number,
    Boolean,
    Date,
    Enum(Vec<String>),
    Array(Box<FieldType>),
    Object(HashMap<String, FieldDefinition>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: ParameterType,
    pub description: String,
    pub default: Option<String>,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ParameterType {
    Number { min: Option<f64>, max: Option<f64> },
    Text { pattern: Option<String> },
    Boolean,
    Enum(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Constraint {
    Length { min: Option<usize>, max: Option<usize> },
    Range { min: Option<f64>, max: Option<f64> },
    Pattern(String),
    Unique,
    Custom(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Relationship {
    pub from_field: String,
    pub to_field: String,
    pub relationship_type: RelationType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RelationType {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeneratedData {
    pub task_name: String,
    pub format: DataFormat,
    pub data: Vec<DataEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DataEntry {
    pub id: String,
    pub content: serde_json::Value,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeneratorResult {
    pub task: String,
    pub format: DataFormat,
    pub count: usize,
    pub data: GeneratedData,
    pub stats: GenerationStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GenerationStats {
    pub generation_time: f64,
    pub success_rate: f64,
    pub error_count: usize,
    pub warnings: Vec<String>,
}