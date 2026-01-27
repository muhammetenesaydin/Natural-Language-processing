# Teknofest Dataset Summary
# This script provides a summary of the most mentioned entities in the dataset

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
    
    for item in dataset:
        if 'entities' in item:
            for entity in item['entities']:
                entity_name = entity['entity']
                sentiment = entity['sentiment']
                entity_sentiments[entity_name].append(sentiment)
    
    return entity_sentiments

def print_summary(entity_sentiments):
    """Print a summary of the most mentioned entities"""
    print("TEKNOFEST DATASET SUMMARY")
    print("=" * 30)
    
    # Calculate statistics
    total_entities = len(entity_sentiments)
    
    # Sort entities by number of mentions
    sorted_entities = sorted(entity_sentiments.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Total unique entities: {total_entities}")
    
    # Show top 10 most mentioned entities
    print(f"\nTop 10 Most Mentioned Entities:")
    print("-" * 40)
    for i, (entity, sentiments) in enumerate(sorted_entities[:10]):
        total = len(sentiments)
        positive = sentiments.count('positive')
        negative = sentiments.count('negative')
        neutral = sentiments.count('neutral')
        
        print(f"{i+1:2d}. {entity}")
        print(f"    Mentions: {total} (Pos: {positive}, Neg: {negative}, Neu: {neutral})")
    
    # Show sentiment distribution
    print(f"\nOverall Sentiment Distribution:")
    print("-" * 40)
    
    total_mentions = sum(len(sentiments) for sentiments in entity_sentiments.values())
    total_positive = sum(sentiments.count('positive') for sentiments in entity_sentiments.values())
    total_negative = sum(sentiments.count('negative') for sentiments in entity_sentiments.values())
    total_neutral = sum(sentiments.count('neutral') for sentiments in entity_sentiments.values())
    
    if total_mentions > 0:
        print(f"Positive: {total_positive} ({total_positive/total_mentions*100:.1f}%)")
        print(f"Negative: {total_negative} ({total_negative/total_mentions*100:.1f}%)")
        print(f"Neutral:  {total_neutral} ({total_neutral/total_mentions*100:.1f}%)")

def main():
    print("Teknofest Dataset Summary")
    print("=" * 25)
    
    # Load dataset
    dataset = load_dataset('veriseti.json')
    
    if dataset:
        print(f"Loaded dataset with {len(dataset)} items")
        
        # Analyze entity sentiments
        entity_sentiments = analyze_entities(dataset)
        
        # Print summary
        print_summary(entity_sentiments)
        
        print("\nSummary complete!")
    else:
        print("Could not load dataset. Please check the file path.")

if __name__ == "__main__":
    main()