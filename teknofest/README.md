# NLP Sentiment Analysis Project

This project demonstrates how to perform sentiment analysis using transformer models in Python.

## Files in this Project

1. **NLP1.ipynb** - Jupyter Notebook with sentiment analysis implementation
2. **simple_torch_sentiment.py** - Simple PyTorch-based sentiment analysis script
3. **veriseti.json** - Dataset file (if available)

## Requirements

- Python 3.7+
- transformers library
- torch (PyTorch)
- datasets (for the notebook)

Install the required packages:
```bash
pip install transformers torch datasets scikit-learn
```

## How to Run

### Option 1: Using the Python Script (Recommended)
```bash
python simple_torch_sentiment.py
```

This script will:
- Download a pre-trained sentiment analysis model
- Analyze sample texts
- Display sentiment predictions with confidence scores

### Option 2: Using the Jupyter Notebook
Open `NLP1.ipynb` in Jupyter Notebook or JupyterLab and run the cells sequentially.

## Troubleshooting

If you encounter issues:

1. **TensorFlow/JAX errors**: The scripts are designed to use PyTorch only. Make sure you have the latest versions of transformers and torch installed.

2. **Model download issues**: Check your internet connection. The first run will download the model files (about 500MB).

3. **Memory issues**: If you have limited RAM, consider using smaller models like `distilbert-base-uncased-finetuned-sst-2-english`.

## Sample Output

The scripts will analyze texts like:
- "I loved this movie!" → Positive (98.75%)
- "This was a terrible film." → Negative (94.64%)

## Customization

To analyze your own texts, modify the `sample_texts` array in the Python script or the `texts` array in the notebook.

## Model Information

The project uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model which is fine-tuned for sentiment analysis and provides three labels:
- Negative
- Neutral  
- Positive