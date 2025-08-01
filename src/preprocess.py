'''
preprocess.py
'''

import pandas as pd

def load_reddit():
    '''
    Returns pd.DataFrame of Reddit post data, adding a new column
    that identifies which ticker the post is commenting on.
    '''

    reddit = pd.read_csv('../data/reddit_wsb.csv')
    reddit['ticker'] = content_to_ticker(reddit['title'])
    return reddit

def content_to_ticker(content):
    '''
    Given the body of a reddit message, identifies which ticker it is
    comments on.
    Parameters:
    content (str): Reddit body message
    Returns:
    str: ticker symbol for the relevant stock
    '''