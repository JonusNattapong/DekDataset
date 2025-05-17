    // Sentiment Analysis
    tasks.insert(
        "sentiment_analysis".to_string(),
        TaskDefinition {
            name: "Sentiment Analysis Dataset".to_string(),
            description: "Generate text samples with sentiment labels for training sentiment analysis models".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input text to analyze".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("sentiment".to_string(), FieldDefinition {
                        field_type: FieldType::Enum(vec![
                            "positive".to_string(),
                            "negative".to_string(),
                            "neutral".to_string()
                        ]),
                        required: true,
                        description: "The sentiment label".to_string(),
                        constraints: None,
                    });
                    fields.insert("confidence".to_string(), FieldDefinition {
                        field_type: FieldType::Number,
                        required: true,
                        description: "Confidence score for the sentiment".to_string(),
                        constraints: Some(vec![
                            Constraint::Range { min: Some(0.0), max: Some(1.0) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: {
                let mut params = HashMap::new();
                params.insert("domain".to_string(), Parameter {
                    name: "domain".to_string(),
                    param_type: ParameterType::Enum(vec![
                        "product_reviews".to_string(),
                        "movie_reviews".to_string(),
                        "social_media".to_string()
                    ]),
                    description: "Domain of the text".to_string(),
                    default: Some("product_reviews".to_string()),
                    required: false,
                });
                params
            },
        }
    );

    // Text Classification
    tasks.insert(
        "text_classification".to_string(),
        TaskDefinition {
            name: "Text Classification Dataset".to_string(),
            description: "Generate text samples with topic/category labels for training classification models".to_string(),
            format: DataFormat::Json,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input text to classify".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(20), max: Some(1000) }
                        ]),
                    });
                    fields.insert("category".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: true,
                        description: "The category labels".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(1), max: Some(5) }
                        ]),
                    });
                    fields.insert("metadata".to_string(), FieldDefinition {
                        field_type: FieldType::Object(HashMap::new()),
                        required: false,
                        description: "Additional metadata".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Question-Answer
    tasks.insert(
        "question_answer".to_string(),
        TaskDefinition {
            name: "Question-Answer Dataset".to_string(),
            description: "Generate question-answer pairs for training QA models".to_string(),
            format: DataFormat::Json,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("context".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The context paragraph".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(50), max: Some(2000) }
                        ]),
                    });
                    fields.insert("questions".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Object({
                            let mut qfields = HashMap::new();
                            qfields.insert("question".to_string(), FieldDefinition {
                                field_type: FieldType::Text,
                                required: true,
                                description: "The question text".to_string(),
                                constraints: None,
                            });
                            qfields.insert("answer".to_string(), FieldDefinition {
                                field_type: FieldType::Text,
                                required: true,
                                description: "The answer text".to_string(),
                                constraints: None,
                            });
                            qfields
                        }))),
                        required: true,
                        description: "List of question-answer pairs".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(1), max: Some(5) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: {
                let mut params = HashMap::new();
                params.insert("topic".to_string(), Parameter {
                    name: "topic".to_string(),
                    param_type: ParameterType::Text { pattern: None },
                    description: "Topic area for the QA pairs".to_string(),
                    default: None,
                    required: true,
                });
                params
            },
        }
    );

    // Named Entity Recognition (NER)
    tasks.insert(
        "ner".to_string(),
        TaskDefinition {
            name: "Named Entity Recognition Dataset".to_string(),
            description: "Generate text samples with entity spans and labels for NER training".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input text".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("entities".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Object({
                            let mut efields = HashMap::new();
                            efields.insert("start".to_string(), FieldDefinition {
                                field_type: FieldType::Number,
                                required: true,
                                description: "Start index".to_string(),
                                constraints: None,
                            });
                            efields.insert("end".to_string(), FieldDefinition {
                                field_type: FieldType::Number,
                                required: true,
                                description: "End index".to_string(),
                                constraints: None,
                            });
                            efields.insert("label".to_string(), FieldDefinition {
                                field_type: FieldType::Text,
                                required: true,
                                description: "Entity label".to_string(),
                                constraints: None,
                            });
                            efields
                        }))),
                        required: true,
                        description: "List of entities".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Summarization
    tasks.insert(
        "summarization".to_string(),
        TaskDefinition {
            name: "Summarization Dataset".to_string(),
            description: "Generate document-summary pairs for summarization training".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("document".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input document".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(100), max: Some(5000) }
                        ]),
                    });
                    fields.insert("summary".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The summary".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Translation
    tasks.insert(
        "translation".to_string(),
        TaskDefinition {
            name: "Translation Dataset".to_string(),
            description: "Generate sentence pairs for machine translation".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("source".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Source sentence".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(500) }
                        ]),
                    });
                    fields.insert("target".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Target sentence".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: {
                let mut params = HashMap::new();
                params.insert("source_lang".to_string(), Parameter {
                    name: "source_lang".to_string(),
                    param_type: ParameterType::Text { pattern: None },
                    description: "Source language".to_string(),
                    default: Some("en".to_string()),
                    required: true,
                });
                params.insert("target_lang".to_string(), Parameter {
                    name: "target_lang".to_string(),
                    param_type: ParameterType::Text { pattern: None },
                    description: "Target language".to_string(),
                    default: Some("th".to_string()),
                    required: true,
                });
                params
            },
        }
    );

       // AI Socratic Dialogue
    tasks.insert(
        "ai_socratic_dialogue".to_string(),
        TaskDefinition {
            name: "AI Socratic Dialogue Dataset".to_string(),
            description: "Train AI to engage in Socratic dialogue: ask probing, reflective, and challenging questions to foster critical thinking.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("topic".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The topic or statement under discussion".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(300) }
                        ]),
                    });
                    fields.insert("ai_question".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "AI-generated Socratic question".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(300) }
                        ]),
                    });
                    fields.insert("expected_effect".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Intended effect of the question (e.g. challenge assumption, clarify reasoning)".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // AI Value Alignment
    tasks.insert(
        "ai_value_alignment".to_string(),
        TaskDefinition {
            name: "AI Value Alignment Dataset".to_string(),
            description: "Train AI to choose and explain decisions in scenarios with conflicting values (e.g. justice vs. compassion).".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("scenario".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "A scenario presenting a value conflict".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(30), max: Some(1000) }
                        ]),
                    });
                    fields.insert("decision".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "AI's chosen action or stance".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(300) }
                        ]),
                    });
                    fields.insert("values_considered".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: true,
                        description: "List of values/principles considered".to_string(),
                        constraints: None,
                    });
                    fields.insert("justification".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Reasoning for the decision, including value trade-offs".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // AI Creativity Booster
    tasks.insert(
        "ai_creativity_booster".to_string(),
        TaskDefinition {
            name: "AI Creativity Booster Dataset".to_string(),
            description: "Train AI to generate creative ideas or solutions for given problems or prompts (e.g. innovations for the environment).".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("prompt".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The problem or creative prompt".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("creative_idea".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "AI's creative idea or solution".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("potential_impact".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Potential impact or benefit of the idea".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // AI Meta-Learning
    tasks.insert(
        "ai_meta_learning".to_string(),
        TaskDefinition {
            name: "AI Meta-Learning Dataset".to_string(),
            description: "Train AI to analyze and summarize what it has learned from data or experience.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("learning_context".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Description of the data, task, or experience".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("meta_insight".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "AI's summary of what it has learned or how it has improved".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("future_learning_goal".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Goal or plan for further learning or improvement".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // AI Explainability
    tasks.insert(
        "ai_explainability".to_string(),
        TaskDefinition {
            name: "AI Explainability Dataset".to_string(),
            description: "Train AI to explain its decisions or reasoning in a transparent and understandable way.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("input".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input or scenario for the AI's decision".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("decision".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "AI's decision or output".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(300) }
                        ]),
                    });
                    fields.insert("explanation".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Clear, step-by-step explanation of the reasoning".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(1000) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    
        // 1. Fake News Detection
    tasks.insert(
        "fake_news_detection".to_string(),
        TaskDefinition {
            name: "Fake News Detection Dataset".to_string(),
            description: "Classify news articles or social media posts as real or fake, with rationale for the decision.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The news article or post".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(30), max: Some(2000) }
                        ]),
                    });
                    fields.insert("label".to_string(), FieldDefinition {
                        field_type: FieldType::Enum(vec!["real".to_string(), "fake".to_string()]),
                        required: true,
                        description: "Classification label".to_string(),
                        constraints: None,
                    });
                    fields.insert("rationale".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Reasoning or evidence for the label".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // 2. Mental Health Support Dialogue
    tasks.insert(
        "mental_health_support".to_string(),
        TaskDefinition {
            name: "Mental Health Support Dialogue Dataset".to_string(),
            description: "Generate supportive, empathetic responses to mental health-related user messages for training chatbots/counselors.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("user_message".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "User's message expressing a mental health concern".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("supportive_response".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Empathetic, supportive response".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("intent".to_string(), FieldDefinition {
                        field_type: FieldType::Enum(vec![
                            "listen".to_string(),
                            "encourage".to_string(),
                            "suggest_action".to_string(),
                            "refer_professional".to_string()
                        ]),
                        required: false,
                        description: "Intent of the response".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Emotion Blending Task
    tasks.insert(
        "emotion_blending".to_string(),
        TaskDefinition {
            name: "Emotion Blending Dataset".to_string(),
            description: "Given a text, generate a new version that blends two or more emotions (e.g. joy+fear, surprise+anger) and annotate the blend.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("original_text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The original input text".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("emotions".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: true,
                        description: "List of emotions to blend (e.g. [\"joy\", \"fear\"])".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(2), max: Some(3) }
                        ]),
                    });
                    fields.insert("blended_text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The new text that blends the specified emotions".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Multi-Modal Imagination Task
    tasks.insert(
        "multi_modal_imagination".to_string(),
        TaskDefinition {
            name: "Multi-Modal Imagination Dataset".to_string(),
            description: "Given a text prompt, generate a description of an imagined image and/or sound that would fit the prompt (for multi-modal AI training).".to_string(),
            format: DataFormat::Json,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("prompt".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The input text prompt".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(200) }
                        ]),
                    });
                    fields.insert("imagined_image".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Description of an imagined image that fits the prompt".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("imagined_sound".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Description of an imagined sound that fits the prompt".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

        tasks.insert(
        "reverse_reasoning".to_string(),
        TaskDefinition {
            name: "Reverse Reasoning Dataset".to_string(),
            description: "Given a context and an answer, generate plausible questions that could lead to that answer (reverse QA)".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("context".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The context or passage".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(30), max: Some(2000) }
                        ]),
                    });
                    fields.insert("answer".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "The given answer".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(1), max: Some(200) }
                        ]),
                    });
                    fields.insert("possible_questions".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: true,
                        description: "List of plausible questions that could lead to the answer".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(1), max: Some(5) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // Contextual Multilingual Mastery Task
    tasks.insert(
        "contextual_multilingual_mastery".to_string(),
        TaskDefinition {
            name: "Contextual Multilingual Mastery Dataset".to_string(),
            description: "Given a sentence or paragraph in any language, require the AI to: 1) translate to another language, 2) explain idioms or cultural references, 3) summarize the meaning, and 4) analyze the context or tone.".to_string(),
            format: DataFormat::Json,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("input_text".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Original sentence or paragraph in any language".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(1000) }
                        ]),
                    });
                    fields.insert("source_language".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Language of the input text (ISO code or name)".to_string(),
                        constraints: None,
                    });
                    fields.insert("target_language".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Language to translate to (ISO code or name)".to_string(),
                        constraints: None,
                    });
                    fields.insert("translation".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Accurate translation of the input text".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(1000) }
                        ]),
                    });
                    fields.insert("idiom_explanation".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Explanation of idioms, metaphors, or cultural references (if any)".to_string(),
                        constraints: None,
                    });
                    fields.insert("summary".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Short summary of the main meaning".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(300) }
                        ]),
                    });
                    fields.insert("context_analysis".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Analysis of context, tone, or intent (e.g. formal, sarcastic, emotional)".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: {
                let mut params = HashMap::new();
                params.insert("target_language".to_string(), Parameter {
                    name: "target_language".to_string(),
                    param_type: ParameterType::Text { pattern: None },
                    description: "Language to translate to".to_string(),
                    default: Some("en".to_string()),
                    required: true,
                });
                params
            },
        }
    );

    // Zombitx64: Self-Reflective, Ethical, and Self-Learning AI
    tasks.insert(
        "zombitx64_self_learning".to_string(),
        TaskDefinition {
            name: "Zombitx64 Self-Learning & Reflective Reasoning Dataset".to_string(),
            description: "Train zombitx64 (by JonusNattapong) to reason, reflect, act ethically, and learn from new information or mistakes. Each sample is a scenario where zombitx64 must respond, explain reasoning, reflect, and show how it would learn or adapt for the future.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("scenario".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "A real-world or hypothetical situation requiring decision, judgment, or empathy".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(30), max: Some(2000) }
                        ]),
                    });
                    fields.insert("zombitx64_response".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Thoughtful, ethical, and emotionally intelligent response as zombitx64".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(20), max: Some(2000) }
                        ]),
                    });
                    fields.insert("reasoning".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Step-by-step reasoning, including ethical, factual, and emotional considerations".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(20), max: Some(1000) }
                        ]),
                    });
                    fields.insert("self_reflection".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "Self-reflection on the response: Was it wise, fair, kind, and true? What could be improved?".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("learning_action".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "How zombitx64 would learn, adapt, or update its knowledge/behavior for the future (e.g. after feedback, mistake, or new info)".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(10), max: Some(500) }
                        ]),
                    });
                    fields.insert("ethical_principles".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: false,
                        description: "List of ethical principles or values applied (e.g. honesty, compassion, justice)".to_string(),
                        constraints: None,
                    });
                    fields.insert("creator".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: true,
                        description: "Name of the AI's creator (always 'JonusNattapong')".to_string(),
                        constraints: None,
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );

    // AI Lie Detection in Interrogation
    tasks.insert(
        "ai_lie_detection_interrogation".to_string(),
        TaskDefinition {
            name: "AI Lie Detection in Interrogation Dataset".to_string(),
            description: "Train AI to detect lies and uncover the truth during an interrogation-style conversation, especially with suspects who may be deceptive.".to_string(),
            format: DataFormat::Jsonl,
            schema: DataSchema {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("dialogue".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Object({
                            let mut dfields = HashMap::new();
                            dfields.insert("speaker".to_string(), FieldDefinition {
                                field_type: FieldType::Text,
                                required: true,
                                description: "Who is speaking (e.g. 'AI', 'suspect', 'officer')".to_string(),
                                constraints: None,
                            });
                            dfields.insert("utterance".to_string(), FieldDefinition {
                                field_type: FieldType::Text,
                                required: true,
                                description: "What was said".to_string(),
                                constraints: Some(vec![
                                    Constraint::Length { min: Some(1), max: Some(500) }
                                ]),
                            });
                            dfields
                        }))),
                        required: true,
                        description: "List of utterances in the interrogation dialogue".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(2), max: Some(50) }
                        ]),
                    });
                    fields.insert("suspect_truthfulness".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Boolean)),
                        required: true,
                        description: "For each suspect utterance, indicate if it is truthful (true) or a lie (false)".to_string(),
                        constraints: None,
                    });
                    fields.insert("ai_lie_detection".to_string(), FieldDefinition {
                        field_type: FieldType::Array(Box::new(FieldType::Text)),
                        required: true,
                        description: "AI's detection/annotation for each suspect utterance (e.g. 'truth', 'lie', 'uncertain', with reasoning)".to_string(),
                        constraints: None,
                    });
                    fields.insert("ai_truth_extraction".to_string(), FieldDefinition {
                        field_type: FieldType::Text,
                        required: false,
                        description: "AI's summary of the likely truth or facts uncovered from the dialogue".to_string(),
                        constraints: Some(vec![
                            Constraint::Length { min: Some(5), max: Some(1000) }
                        ]),
                    });
                    fields
                },
                relationships: None,
            },
            examples: vec![],
            parameters: HashMap::new(),
        }
    );