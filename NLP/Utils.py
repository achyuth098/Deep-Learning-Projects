import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# ensure NLTK corpora are present
for pkg in ("punkt","wordnet","omw-1.4"):
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
vader    = SentimentIntensityAnalyzer()

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Review','Rating']]
    df['Sentiment'] = df['Rating'].apply(lambda x: 'Positive' if x>=3 else 'Negative')
    return df

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)                # strip HTML
    tokens = word_tokenize(text)                     # tokenize
    tokens = [t.lower() for t in tokens if t.isalpha()]  # keep words only
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def textblob_sentiment(text):
    return 'Positive' if TextBlob(text).sentiment.polarity > 0 else 'Negative'

def vader_sentiment(text):
    return 'Positive' if vader.polarity_scores(text)['compound'] > 0 else 'Negative'
