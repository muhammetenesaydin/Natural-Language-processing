# Simple Teknofest Dataset Analysis
# This script analyzes the Teknofest dataset

import json
from collections import defaultdict

def load_dataset(file_path):
    """Load the Teknofest dataset from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def analyze_entities(dataset):
    """Analyze entity sentiments in the dataset"""
    # Track entity sentiments
    entity_sentiments = defaultdict(list)
    
    print("Analyzing entity sentiments in the dataset...")
    print(f"Processing {len(dataset)} items...")
    
    for item in dataset:
        if 'entities' in item:
            for entity in item['entities']:
                entity_name = entity['entity']
                sentiment = entity['sentiment']
                entity_sentiments[entity_name].append(sentiment)
    
    return entity_sentiments

def print_analysis(entity_sentiments):
    """Print analysis of entity sentiments"""
    print("\n" + "="*50)
    print("TEKNOFEST DATASET ANALYSIS")
    print("="*50)
    
    # Sort entities by number of mentions
    sorted_entities = sorted(entity_sentiments.items(), key=lambda x: len(x[1]), reverse=True)
    
    for entity, sentiments in sorted_entities:
        total = len(sentiments)
        positive = sentiments.count('positive')
        negative = sentiments.count('negative')
        neutral = sentiments.count('neutral')
        
        print(f"\n{entity}:")
        print(f"  Total mentions: {total}")
        if total > 0:
            print(f"  Positive: {positive} ({positive/total*100:.1f}%)")
            print(f"  Negative: {negative} ({negative/total*100:.1f}%)")
            print(f"  Neutral: {neutral} ({neutral/total*100:.1f}%)")

def main():
    print("Teknofest Dataset Analysis")
    print("=" * 30)
    
    # Load dataset
    dataset = load_dataset('veriseti.json')
    
    if dataset:
        print(f"Loaded dataset with {len(dataset)} items")
        
        # Analyze entity sentiments
        entity_sentiments = analyze_entities(dataset)
        
        # Print analysis
        print_analysis(entity_sentiments)
        
        print(f"\nFound {len(entity_sentiments)} unique entities in the dataset.")
        print("\nAnalysis complete!")
    else:
        print("Could not load dataset. Please check the file path.")

if __name__ == "__main__":
    main()