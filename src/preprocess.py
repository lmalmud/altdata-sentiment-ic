'''
preprocess.py
'''

import pandas as pd
import re # For RegEx parsing

def load_reddit():
    '''
    Returns pd.DataFrame of Reddit post data, adding a new column
    that identifies which ticker the post is commenting on.
    '''

    reddit = pd.read_csv('../data/reddit_wsb.csv')

    # Add one column that will combine the title with the body
    reddit['raw_text'] = reddit['title'].fillna('') + ' ' + reddit['body'].fillna('')
    
    # Apply the funciton that will extract the list of valid tickers
    reddit['ticker'] = reddit['raw_text'].apply(valid_mentions)

    # Keep posts with >= 1 ticker
    reddit = reddit[reddit['ticker'].str.len() > 0]

    reddit.to_csv('../data/reddit_wsb_parsed.csv', index=False)
    
    return reddit

def load_symbols():
    '''
    Returns a list of all symbols and names included in the dataset.
    Returns:
    List[str]: symbols
    List[str]: descriptions
    '''
    syms = pd.read_csv('../data/symbols_valid_meta.csv')
    return list(syms['Symbol'])

# RegEx ticker identifier
TICKER_RE = re.compile(r"""
    (?<![A-Za-z])               # left boundary not a letter
    (?:\$)?                     # optional leading $
    ([A-Z]{2,5})                # 2–5 capital letters
    (?![A-Za-z])                # right boundary not a letter
    """, re.VERBOSE)

# Words that may be identified as tickers but should not be
STOP_WORDS = {"YOLO", "USA", "METH", "DD", "ATM", "HODL"}

# Tickers we're looking for
valid_tickers = load_symbols()

def extract_candidates(text: str) -> set[str]:
    '''
    Extracts all potential ticker symbols from text using RegEx.
    Parameters:
    text (str): text to identify tickers from
    Returns:
    set (str): of identified tickers
    '''

    # text or "" handles the case where text might be None by default to empty string
    # used the regex pattern TICKER_RE
    # m.group(1) will extract jsut the ticker letters (group 1), ignoring potential $
    return {m.group(1) for m in TICKER_RE.finditer(text or "")}

def valid_mentions(text: str) -> set[str]:
    '''
    Filters candidate tickers to only include legitimate symbols.
    Parameters:
    text (str): text to be parsed
    Returns:
    set (str): valid tickers
    '''
    raw = extract_candidates(text) # Extract the candidates using helper

    # Only return the symbols that are not in STOP_WORDS
    return {t for t in raw if t in valid_tickers and t not in STOP_WORDS}

load_reddit()