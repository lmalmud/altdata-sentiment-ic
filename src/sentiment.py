'''
sentiment.py
Uses preprocessed file and performs sentiment analysis on the raw text.

Source:
https://medium.com/@ravirajshinde2000/financial-news-sentiment-analysis-using-finbert-25afcc95e65f
'''

from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import torch

# Load pre-trained FinBERT model and tokenizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

reddit_parsed = pd.read_csv('../data/reddit_wsb_parsed.csv')

# labels = {0:'neutral', 1:'positive',2:'negative'}
reorderings = {0: 1, 1: 0, 2: 2} # so rankings are positive -> negative

def get_sentiment(txt):
    '''
    Given a text snippet, returns the predicted sentiment score
    using the FinBERT model.
    Parameters:
    txt (string): text to be analysed
    Returns:

    '''

    # Inputs is a dictionary containing the tokenized and encoded version of text
    inputs = tokenizer(txt,
                       return_tensors='pt',
                       padding=True,
                       truncation=True, # Handles long texts
                       max_length=512)
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = finbert(**inputs)
        logits = outputs.logits
        # Logits are raw, unnormalized output scores before any final processing
        # Will be something like [[-.5, 2.1, -1.3]], representing the scores for index 0 (neutral)
        # index 1 (positive), and index 2 (negative) -- we want the highest
        # The logits are not probabilities
        predicted_class = torch.argmax(logits, dim=1).item() # Gets the predicted class as integer
    
    return predicted_class

def convert_sentiment(x):
    return reorderings[x]

reddit_parsed['sentiment'] = reddit_parsed['raw_text'].apply(get_sentiment)

# Convert the scores to appropriate numerical values
reddit_parsed['sentiment'].apply(convert_sentiment)

reddit_parsed.to_csv('../data/reddit_wsb_with_sentiment.csv', index=False)