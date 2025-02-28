#!/usr/bin/env python3
"""
Script to fine-tune the XLM-RoBERTa model on the MultiWOZ dataset.
This script should be run separately as it requires significant computational resources.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the ChatbotModel
from api.model import ChatbotModel

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune the XLM-RoBERTa model on MultiWOZ dataset')
    parser.add_argument('--model', type=str, default=os.getenv('MODEL_NAME', 'FacebookAI/xlm-roberta-large'),
                        help='Model name or path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--output-dir', type=str, default='models/fine_tuned_model',
                        help='Output directory for the fine-tuned model')
    
    args = parser.parse_args()
    
    print(f"Fine-tuning model: {args.model}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize the model
    model = ChatbotModel()
    
    # Fine-tune the model
    model.fine_tune()
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()