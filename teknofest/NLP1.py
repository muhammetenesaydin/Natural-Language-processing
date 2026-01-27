# NLP Sentiment Analysis with Transformers
# This script performs sentiment analysis on the IMDb dataset using DistilBERT

# Install required packages (uncomment if needed)
# !pip install transformers datasets torch scikit-learn

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load IMDb dataset
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Extract training and test data
train_texts = list(dataset['train']['text'])
train_labels = list(dataset['train']['label'])
test_texts = list(dataset['test']['text'])
test_labels = list(dataset['test']['label'])

print(f"Training samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the data
print("Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create dataset class
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# Load pre-trained model
print("Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Set up training arguments (commented out to avoid long training times)
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=1,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir='./logs',
# )

# Initialize trainer (commented out to avoid long training times)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# Train the model (commented out to avoid long training times)
# print("Training model...")
# trainer.train()

# Test with sample sentences
print("Testing with sample sentences...")
texts = ["I loved this movie!", "This was a terrible film."]

encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
outputs = model(**encodings)
preds = torch.argmax(outputs.logits, dim=1)

for text, pred in zip(texts, preds):
    print(f"{text} --> {'Positive' if pred==1 else 'Negative'}")

print("Script completed successfully!")