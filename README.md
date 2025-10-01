# Sentiment Analysis with Pre-trained Models

This project implements sentiment analysis using transformer models to classify text as positive, negative, or neutral. It includes four main scripts:

1. `pretrained_sentiment.py` - Uses a pre-trained sentiment analysis model (recommended for immediate use)
2. `simple_sentiment.py` - Uses DistilBERT with random weights (for demonstration)
3. `final_train.py` - A simplified training script with manual training loop (recommended for learning)
4. `complete_workflow.py` - A complete end-to-end workflow demonstrating training, saving, and inference

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Accelerate (for training)

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

If you encounter issues with TensorFlow dependencies, you can install the packages individually:
```bash
pip install torch transformers datasets scikit-learn accelerate
```

## Usage

### Pre-trained Model (Recommended)

Run the pre-trained sentiment analysis script:
```bash
python pretrained_sentiment.py
```

This script will:
1. Load a pre-trained sentiment analysis model
2. Analyze sample sentences for sentiment
3. Provide accurate sentiment predictions

### Complete Workflow (Recommended for Learning)

Run the complete workflow script:
```bash
python complete_workflow.py
```

This script demonstrates the complete end-to-end workflow:
1. Training a model on sample data
2. Saving the trained model
3. Loading the trained model
4. Performing inference on new data

### Simple Inference (Educational)

Run the lightweight sentiment analysis script:
```bash
python simple_sentiment.py
```

This script will:
1. Load the DistilBERT model and tokenizer
2. Analyze sample sentences for sentiment

Note: This uses an untrained model with random weights, so predictions will be random.

### Training a Model (Educational)

Run the simplified training script (recommended for learning):
```bash
python final_train.py
```

This script will:
1. Create sample training data
2. Tokenize the data using DistilBERT tokenizer
3. Load the pre-trained DistilBERT model
4. Train the model using a manual training loop
5. Evaluate the model during training
6. Save the trained model to disk

Note: This is a simplified example for learning purposes.

## Customization

To train the model on the IMDb dataset, uncomment the training section in the `main()` function:
```python
# Train model (commented out as it takes time)
trainer = train_model(model, train_dataset, test_dataset)
```

Note that training can take several hours depending on your hardware.

## Sample Output

The pre-trained sentiment script will output accurate sentiment predictions for sample sentences:
```
Positive   | I loved this movie! It was fantastic.
Negative   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Negative   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```

The simple sentiment script (using untrained model) will output random predictions:
```
Positive   | I loved this movie! It was fantastic.
Positive   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Positive   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```