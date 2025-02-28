#!/usr/bin/env python3
"""
Script to download and preprocess the MultiWOZ dataset.
"""

import os
import sys
from datasets import load_dataset
import json
from tqdm import tqdm
from pathlib import Path

def main():
    print("Downloading MultiWOZ dataset...")
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    try:
        # Load the dataset
        dataset = load_dataset("multi_woz_v22")
        
        # Save dataset statistics
        stats = {
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset["validation"]),
            "test_size": len(dataset["test"]),
            "domains": dataset["train"].features["dialogue"][0]["domain"].names,
            "intents": dataset["train"].features["dialogue"][0]["intent"].names
        }
        
        with open("data/dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("Dataset statistics:")
        print(f"  Train size: {stats['train_size']}")
        print(f"  Validation size: {stats['validation_size']}")
        print(f"  Test size: {stats['test_size']}")
        print(f"  Domains: {', '.join(stats['domains'])}")
        print(f"  Intents: {', '.join(stats['intents'])}")
        
        # Extract sample dialogues
        sample_dialogues = []
        for i, dialogue in enumerate(dataset["train"]):
            if i >= 5:  # Just get 5 samples
                break
            sample_dialogues.append(dialogue)
        
        with open("data/sample_dialogues.json", "w") as f:
            json.dump(sample_dialogues, f, indent=2)
        
        print("MultiWOZ dataset downloaded and processed successfully!")
        print("Sample dialogues saved to data/sample_dialogues.json")
        print("Dataset statistics saved to data/dataset_stats.json")
    
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()