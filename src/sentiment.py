'''
sentiment.py
Uses preprocessed file and performs sentiment analysis on the raw text.

Source:
https://medium.com/@ravirajshinde2000/financial-news-sentiment-analysis-using-finbert-25afcc95e65f
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np
import torch

# Load pre-trained FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

reddit_parsed = pd.read_csv('../data/reddit_wsb_parsed.csv')

def get_sentiment(txt):
    '''
    Given a text snippet, returns the predicted sentiment score
    using the FinBERT model.
    Parameters:
    txt (string): text to be analysed
    Returns:

    '''

    # encoded_input is a dictionary containing the tokenized and encoded version of text
    # Need to include truncation and max_length, otherwise model does not work
    encoded_input = tokenizer(txt,
                            return_tensors='pt',
                            truncation=True, 
                            max_length=512)
    
    # Get the model output
    output = model(**encoded_input)

    # Apply softmax to get probabilities
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Calculate the compound score
    # The order of labels is [positive, negative, neutral]
    compound_score = scores[0] - scores[1]  # Positive - Negative
    return compound_score


reddit_parsed['sentiment'] = reddit_parsed['raw_text'].apply(get_sentiment)

reddit_parsed.to_csv('../data/reddit_wsb_with_sentiment.csv', index=False)

# TESTS
'''
text = "The company's stock price has surged recently due to positive earnings reports."
compound_score = get_sentiment(text)
print(f"FinBERT Compound Score: {compound_score}")

text2 = "The company's financial outlook is bleak with declining revenues."
compound_score2 = get_sentiment(text2)
print(f"FinBERT Compound Score: {compound_score2}")
'''