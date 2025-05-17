# task_definitions.py
# รวม task definition ตาม schema Rust (ตัวอย่าง 6 task หลัก)

from typing import Any, Dict, List, Optional, Union

def get_task_definitions() -> Dict[str, Any]:
    # All task definitions as Python dicts, not Rust code
    return {
        "sentiment_analysis": {
            "name": "Sentiment Analysis Dataset",
            "description": "Generate text samples with sentiment labels for training sentiment analysis models",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input text to analyze",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "sentiment": {
                        "field_type": "enum",
                        "enum_values": ["positive", "negative", "neutral"],
                        "required": True,
                        "description": "The sentiment label",
                    },
                    "confidence": {
                        "field_type": "number",
                        "required": True,
                        "description": "Confidence score for the sentiment",
                        "constraints": {"range": {"min": 0.0, "max": 1.0}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {
                "domain": {
                    "name": "domain",
                    "param_type": "enum",
                    "enum_values": ["product_reviews", "movie_reviews", "social_media"],
                    "description": "Domain of the text",
                    "default": "product_reviews",
                    "required": False,
                }
            },
        },
        "text_classification": {
            "name": "Text Classification Dataset",
            "description": "Generate text samples with topic/category labels for training classification models",
            "format": "json",
            "schema": {
                "fields": {
                    "text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input text to classify",
                        "constraints": {"length": {"min": 20, "max": 1000}},
                    },
                    "category": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": True,
                        "description": "The category labels",
                        "constraints": {"length": {"min": 1, "max": 5}},
                    },
                    "metadata": {
                        "field_type": "object",
                        "required": False,
                        "description": "Additional metadata",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "question_answer": {
            "name": "Question-Answer Dataset",
            "description": "Generate question-answer pairs for training QA models",
            "format": "json",
            "schema": {
                "fields": {
                    "context": {
                        "field_type": "text",
                        "required": True,
                        "description": "The context paragraph",
                        "constraints": {"length": {"min": 50, "max": 2000}},
                    },
                    "questions": {
                        "field_type": "array",
                        "items": {
                            "field_type": "object",
                            "fields": {
                                "question": {
                                    "field_type": "text",
                                    "required": True,
                                    "description": "The question text",
                                },
                                "answer": {
                                    "field_type": "text",
                                    "required": True,
                                    "description": "The answer text",
                                },
                            },
                        },
                        "required": True,
                        "description": "List of question-answer pairs",
                        "constraints": {"length": {"min": 1, "max": 5}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {
                "topic": {
                    "name": "topic",
                    "param_type": "text",
                    "description": "Topic area for the QA pairs",
                    "required": True,
                }
            },
        },
        "ner": {
            "name": "Named Entity Recognition Dataset",
            "description": "Generate text samples with entity spans and labels for NER training",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input text",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "entities": {
                        "field_type": "array",
                        "items": {
                            "field_type": "object",
                            "fields": {
                                "start": {
                                    "field_type": "number",
                                    "required": True,
                                    "description": "Start index",
                                },
                                "end": {
                                    "field_type": "number",
                                    "required": True,
                                    "description": "End index",
                                },
                                "label": {
                                    "field_type": "text",
                                    "required": True,
                                    "description": "Entity label",
                                },
                            },
                        },
                        "required": True,
                        "description": "List of entities",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "summarization": {
            "name": "Summarization Dataset",
            "description": "Generate document-summary pairs for summarization training",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "document": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input document",
                        "constraints": {"length": {"min": 100, "max": 5000}},
                    },
                    "summary": {
                        "field_type": "text",
                        "required": True,
                        "description": "The summary",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "translation": {
            "name": "Translation Dataset",
            "description": "Generate sentence pairs for machine translation",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "source": {
                        "field_type": "text",
                        "required": True,
                        "description": "Source sentence",
                        "constraints": {"length": {"min": 5, "max": 500}},
                    },
                    "target": {
                        "field_type": "text",
                        "required": True,
                        "description": "Target sentence",
                        "constraints": {"length": {"min": 5, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {
                "source_lang": {
                    "name": "source_lang",
                    "param_type": "text",
                    "description": "Source language",
                    "default": "en",
                    "required": True,
                },
                "target_lang": {
                    "name": "target_lang",
                    "param_type": "text",
                    "description": "Target language",
                    "default": "th",
                    "required": True,
                },
            },
        },
        "ai_socratic_dialogue": {
            "name": "AI Socratic Dialogue Dataset",
            "description": "Train AI to engage in Socratic dialogue: ask probing, reflective, and challenging questions to foster critical thinking.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "topic": {
                        "field_type": "text",
                        "required": True,
                        "description": "The topic or statement under discussion",
                        "constraints": {"length": {"min": 10, "max": 300}},
                    },
                    "ai_question": {
                        "field_type": "text",
                        "required": True,
                        "description": "AI-generated Socratic question",
                        "constraints": {"length": {"min": 10, "max": 300}},
                    },
                    "expected_effect": {
                        "field_type": "text",
                        "required": False,
                        "description": "Intended effect of the question (e.g. challenge assumption, clarify reasoning)",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "ai_value_alignment": {
            "name": "AI Value Alignment Dataset",
            "description": "Train AI to choose and explain decisions in scenarios with conflicting values (e.g. justice vs. compassion).",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "scenario": {
                        "field_type": "text",
                        "required": True,
                        "description": "A scenario presenting a value conflict",
                        "constraints": {"length": {"min": 30, "max": 1000}},
                    },
                    "decision": {
                        "field_type": "text",
                        "required": True,
                        "description": "AI's chosen action or stance",
                        "constraints": {"length": {"min": 5, "max": 300}},
                    },
                    "values_considered": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": True,
                        "description": "List of values/principles considered",
                    },
                    "justification": {
                        "field_type": "text",
                        "required": True,
                        "description": "Reasoning for the decision, including value trade-offs",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "ai_creativity_booster": {
            "name": "AI Creativity Booster Dataset",
            "description": "Train AI to generate creative ideas or solutions for given problems or prompts (e.g. innovations for the environment).",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "prompt": {
                        "field_type": "text",
                        "required": True,
                        "description": "The problem or creative prompt",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "creative_idea": {
                        "field_type": "text",
                        "required": True,
                        "description": "AI's creative idea or solution",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "potential_impact": {
                        "field_type": "text",
                        "required": False,
                        "description": "Potential impact or benefit of the idea",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "ai_meta_learning": {
            "name": "AI Meta-Learning Dataset",
            "description": "Train AI to analyze and summarize what it has learned from data or experience.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "learning_context": {
                        "field_type": "text",
                        "required": True,
                        "description": "Description of the data, task, or experience",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "meta_insight": {
                        "field_type": "text",
                        "required": True,
                        "description": "AI's summary of what it has learned or how it has improved",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "future_learning_goal": {
                        "field_type": "text",
                        "required": False,
                        "description": "Goal or plan for further learning or improvement",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "ai_explainability": {
            "name": "AI Explainability Dataset",
            "description": "Train AI to explain its decisions or reasoning in a transparent and understandable way.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "input": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input or scenario for the AI's decision",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "decision": {
                        "field_type": "text",
                        "required": True,
                        "description": "AI's decision or output",
                        "constraints": {"length": {"min": 5, "max": 300}},
                    },
                    "explanation": {
                        "field_type": "text",
                        "required": True,
                        "description": "Clear, step-by-step explanation of the reasoning",
                        "constraints": {"length": {"min": 10, "max": 1000}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "fake_news_detection": {
            "name": "Fake News Detection Dataset",
            "description": "Classify news articles or social media posts as real or fake, with rationale for the decision.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The news article or post",
                        "constraints": {"length": {"min": 30, "max": 2000}},
                    },
                    "label": {
                        "field_type": "enum",
                        "enum_values": ["real", "fake"],
                        "required": True,
                        "description": "Classification label",
                    },
                    "rationale": {
                        "field_type": "text",
                        "required": False,
                        "description": "Reasoning or evidence for the label",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "mental_health_support": {
            "name": "Mental Health Support Dialogue Dataset",
            "description": "Generate supportive, empathetic responses to mental health-related user messages for training chatbots/counselors.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "user_message": {
                        "field_type": "text",
                        "required": True,
                        "description": "User's message expressing a mental health concern",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "supportive_response": {
                        "field_type": "text",
                        "required": True,
                        "description": "Empathetic, supportive response",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "intent": {
                        "field_type": "enum",
                        "enum_values": ["listen", "encourage", "suggest_action", "refer_professional"],
                        "required": False,
                        "description": "Intent of the response",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "emotion_blending": {
            "name": "Emotion Blending Dataset",
            "description": "Given a text, generate a new version that blends two or more emotions (e.g. joy+fear, surprise+anger) and annotate the blend.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "original_text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The original input text",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "emotions": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": True,
                        "description": "List of emotions to blend (e.g. [\"joy\", \"fear\"])",
                        "constraints": {"length": {"min": 2, "max": 3}},
                    },
                    "blended_text": {
                        "field_type": "text",
                        "required": True,
                        "description": "The new text that blends the specified emotions",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "multi_modal_imagination": {
            "name": "Multi-Modal Imagination Dataset",
            "description": "Given a text prompt, generate a description of an imagined image and/or sound that would fit the prompt (for multi-modal AI training).",
            "format": "json",
            "schema": {
                "fields": {
                    "prompt": {
                        "field_type": "text",
                        "required": True,
                        "description": "The input text prompt",
                        "constraints": {"length": {"min": 5, "max": 200}},
                    },
                    "imagined_image": {
                        "field_type": "text",
                        "required": False,
                        "description": "Description of an imagined image that fits the prompt",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "imagined_sound": {
                        "field_type": "text",
                        "required": False,
                        "description": "Description of an imagined sound that fits the prompt",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "reverse_reasoning": {
            "name": "Reverse Reasoning Dataset",
            "description": "Given a context and an answer, generate plausible questions that could lead to that answer (reverse QA)",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "context": {
                        "field_type": "text",
                        "required": True,
                        "description": "The context or passage",
                        "constraints": {"length": {"min": 30, "max": 2000}},
                    },
                    "answer": {
                        "field_type": "text",
                        "required": True,
                        "description": "The given answer",
                        "constraints": {"length": {"min": 1, "max": 200}},
                    },
                    "possible_questions": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": True,
                        "description": "List of plausible questions that could lead to the answer",
                        "constraints": {"length": {"min": 1, "max": 5}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "contextual_multilingual_mastery": {
            "name": "Contextual Multilingual Mastery Dataset",
            "description": "Given a sentence or paragraph in any language, require the AI to: 1) translate to another language, 2) explain idioms or cultural references, 3) summarize the meaning, and 4) analyze the context or tone.",
            "format": "json",
            "schema": {
                "fields": {
                    "input_text": {
                        "field_type": "text",
                        "required": True,
                        "description": "Original sentence or paragraph in any language",
                        "constraints": {"length": {"min": 10, "max": 1000}},
                    },
                    "source_language": {
                        "field_type": "text",
                        "required": True,
                        "description": "Language of the input text (ISO code or name)",
                    },
                    "target_language": {
                        "field_type": "text",
                        "required": True,
                        "description": "Language to translate to (ISO code or name)",
                    },
                    "translation": {
                        "field_type": "text",
                        "required": True,
                        "description": "Accurate translation of the input text",
                        "constraints": {"length": {"min": 10, "max": 1000}},
                    },
                    "idiom_explanation": {
                        "field_type": "text",
                        "required": False,
                        "description": "Explanation of idioms, metaphors, or cultural references (if any)",
                    },
                    "summary": {
                        "field_type": "text",
                        "required": False,
                        "description": "Short summary of the main meaning",
                        "constraints": {"length": {"min": 5, "max": 300}},
                    },
                    "context_analysis": {
                        "field_type": "text",
                        "required": False,
                        "description": "Analysis of context, tone, or intent (e.g. formal, sarcastic, emotional)",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {
                "target_language": {
                    "name": "target_language",
                    "param_type": "text",
                    "description": "Language to translate to",
                    "default": "en",
                    "required": True,
                }
            },
        },
        "zombitx64_self_learning": {
            "name": "Zombitx64 Self-Learning & Reflective Reasoning Dataset",
            "description": "Train zombitx64 (by JonusNattapong) to reason, reflect, act ethically, and learn from new information or mistakes. Each sample is a scenario where zombitx64 must respond, explain reasoning, reflect, and show how it would learn or adapt for the future.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "scenario": {
                        "field_type": "text",
                        "required": True,
                        "description": "A real-world or hypothetical situation requiring decision, judgment, or empathy",
                        "constraints": {"length": {"min": 30, "max": 2000}},
                    },
                    "zombitx64_response": {
                        "field_type": "text",
                        "required": True,
                        "description": "Thoughtful, ethical, and emotionally intelligent response as zombitx64",
                        "constraints": {"length": {"min": 20, "max": 2000}},
                    },
                    "reasoning": {
                        "field_type": "text",
                        "required": True,
                        "description": "Step-by-step reasoning, including ethical, factual, and emotional considerations",
                        "constraints": {"length": {"min": 20, "max": 1000}},
                    },
                    "self_reflection": {
                        "field_type": "text",
                        "required": False,
                        "description": "Self-reflection on the response: Was it wise, fair, kind, and true? What could be improved?",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "learning_action": {
                        "field_type": "text",
                        "required": False,
                        "description": "How zombitx64 would learn, adapt, or update its knowledge/behavior for the future (e.g. after feedback, mistake, or new info)",
                        "constraints": {"length": {"min": 10, "max": 500}},
                    },
                    "ethical_principles": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": False,
                        "description": "List of ethical principles or values applied (e.g. honesty, compassion, justice)",
                    },
                    "creator": {
                        "field_type": "text",
                        "required": True,
                        "description": "Name of the AI's creator (always 'JonusNattapong')",
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
        "ai_lie_detection_interrogation": {
            "name": "AI Lie Detection in Interrogation Dataset",
            "description": "Train AI to detect lies and uncover the truth during an interrogation-style conversation, especially with suspects who may be deceptive.",
            "format": "jsonl",
            "schema": {
                "fields": {
                    "dialogue": {
                        "field_type": "array",
                        "items": {
                            "field_type": "object",
                            "fields": {
                                "speaker": {
                                    "field_type": "text",
                                    "required": True,
                                    "description": "Who is speaking (e.g. 'AI', 'suspect', 'officer')",
                                },
                                "utterance": {
                                    "field_type": "text",
                                    "required": True,
                                    "description": "What was said",
                                    "constraints": {"length": {"min": 1, "max": 500}},
                                },
                            },
                        },
                        "required": True,
                        "description": "List of utterances in the interrogation dialogue",
                        "constraints": {"length": {"min": 2, "max": 50}},
                    },
                    "suspect_truthfulness": {
                        "field_type": "array",
                        "items": {"field_type": "boolean"},
                        "required": True,
                        "description": "For each suspect utterance, indicate if it is truthful (true) or a lie (false)",
                    },
                    "ai_lie_detection": {
                        "field_type": "array",
                        "items": {"field_type": "text"},
                        "required": True,
                        "description": "AI's detection/annotation for each suspect utterance (e.g. 'truth', 'lie', 'uncertain', with reasoning)",
                    },
                    "ai_truth_extraction": {
                        "field_type": "text",
                        "required": False,
                        "description": "AI's summary of the likely truth or facts uncovered from the dialogue",
                        "constraints": {"length": {"min": 5, "max": 1000}},
                    },
                },
                "relationships": None,
            },
            "examples": [],
            "parameters": {},
        },
    }
