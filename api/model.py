import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
import json
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

class ChatbotModel:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'FacebookAI/xlm-roberta-large')
        self.models = {}
        self.tokenizers = {}
        self.fine_tuned = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
        # Check if fine-tuned model exists
        if Path("models/fine_tuned_model").exists():
            self.load_fine_tuned_model()
            self.fine_tuned = True
        else:
            print("No fine-tuned model found. Using base model.")
            self.load_base_model()
    
    def load_base_model(self):
        """Load the base model and tokenizer"""
        print(f"Loading base model: {self.model_name}")
        try:
            self.tokenizers[self.model_name] = AutoTokenizer.from_pretrained(self.model_name)
            self.models[self.model_name] = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=7  # MultiWOZ has 7 domains
            )
            print("Base model loaded successfully")
        except Exception as e:
            print(f"Error loading base model: {str(e)}")
    
    def load_fine_tuned_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("Loading fine-tuned model")
        try:
            model_path = "models/fine_tuned_model"
            self.tokenizers[self.model_name] = AutoTokenizer.from_pretrained(model_path)
            self.models[self.model_name] = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load response templates
            with open("models/response_templates.json", "r") as f:
                self.response_templates = json.load(f)
            
            print("Fine-tuned model loaded successfully")
        except Exception as e:
            print(f"Error loading fine-tuned model: {str(e)}")
            # Fallback to base model
            self.load_base_model()
    
    def fine_tune(self):
        """Fine-tune the model on MultiWOZ dataset"""
        print("Starting fine-tuning process...")
        
        try:
            # Load MultiWOZ dataset
            dataset = load_dataset("multi_woz_v22")
            
            # Preprocess dataset
            tokenizer = self.tokenizers[self.model_name]
            
            def preprocess_function(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
            
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            
            # Train the model
            from transformers import Trainer, TrainingArguments
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
            )
            
            trainer = Trainer(
                model=self.models[self.model_name],
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"]
            )
            
            trainer.train()
            
            # Save the fine-tuned model
            self.models[self.model_name].save_pretrained("models/fine_tuned_model")
            tokenizer.save_pretrained("models/fine_tuned_model")
            
            # Extract response templates from dataset
            self._extract_response_templates(dataset)
            
            self.fine_tuned = True
            print("Fine-tuning completed successfully")
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
    
    def _extract_response_templates(self, dataset):
        """Extract response templates from the dataset"""
        response_templates = {}
        
        for dialogue in tqdm(dataset["train"]["dialogue"]):
            for turn in dialogue:
                if turn["role"] == "system":
                    domain = turn.get("domain", "general")
                    intent = turn.get("intent", "inform")
                    response = turn.get("text", "")
                    
                    if domain not in response_templates:
                        response_templates[domain] = {}
                    
                    if intent not in response_templates[domain]:
                        response_templates[domain][intent] = []
                    
                    if response and response not in response_templates[domain][intent]:
                        response_templates[domain][intent].append(response)
        
        # Save response templates
        with open("models/response_templates.json", "w") as f:
            json.dump(response_templates, f, indent=2)
    
    def generate_response(self, message, language="en", model_name=None):
        """Generate a response to the user message"""
        if not model_name:
            model_name = self.model_name
        
        # Load model if not already loaded
        if model_name not in self.models:
            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                # Fallback to default model
                model_name = self.model_name
        
        # If not fine-tuned, return a generic response
        if not self.fine_tuned:
            generic_responses = {
                "en": "I'm a customer support chatbot. How can I help you today?",
                "fr": "Je suis un chatbot de support client. Comment puis-je vous aider aujourd'hui?",
                "de": "Ich bin ein Kundensupport-Chatbot. Wie kann ich Ihnen heute helfen?",
                "es": "Soy un chatbot de atención al cliente. ¿Cómo puedo ayudarte hoy?",
                "it": "Sono un chatbot di supporto clienti. Come posso aiutarti oggi?",
                "zh": "我是一个客户支持聊天机器人。今天我能帮您什么忙？",
                "ja": "私はカスタマーサポートチャットボットです。今日はどのようにお手伝いできますか？",
                "ko": "저는 고객 지원 챗봇입니다. 오늘 어떻게 도와 드릴까요?",
                "ar": "أنا روبوت دردشة لدعم العملاء. كيف يمكنني مساعدتك اليوم؟",
                "ru": "Я чат-бот службы поддержки клиентов. Чем я могу помочь вам сегодня?"
            }
            return generic_responses.get(language, generic_responses["en"])
        
        try:
            # Tokenize input
            tokenizer = self.tokenizers[model_name]
            inputs = tokenizer(message, return_tensors="pt")
            
            # Get model prediction
            model = self.models[model_name]
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1).item()
            
            # Map prediction to domain and intent
            domains = ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]
            domain = domains[predictions % len(domains)]
            
            # Get response from templates
            if hasattr(self, 'response_templates') and domain in self.response_templates:
                intents = list(self.response_templates[domain].keys())
                intent = intents[0] if intents else "inform"
                
                if intent in self.response_templates[domain] and self.response_templates[domain][intent]:
                    responses = self.response_templates[domain][intent]
                    response = np.random.choice(responses)
                else:
                    response = f"I can help you with {domain} related queries. What specific information do you need?"
            else:
                response = f"I can help you with {domain} related queries. What specific information do you need?"
            
            # Translate response if needed (using the multilingual capabilities of XLM-RoBERTa)
            if language != "en":
                # For simplicity, we're not implementing actual translation here
                # In a real implementation, you would use a translation model or API
                pass
            
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."