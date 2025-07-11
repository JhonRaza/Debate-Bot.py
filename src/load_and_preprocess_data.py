import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import os

def preprocess_data():
    cols = ['id', 'label', 'statement', 'subject', 'speaker', 'designation', 'state', 
        'party', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 
        'pants_on_fire_counts', 'context']
    
    df = pd.read_csv('data/train.tsv', sep='\t', names=cols)


    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        print("⚠️ Stopwords not found. Downloading...")
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    def clean_data(text):
        text = text.lower()  # Lowercase
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # removing special characters
        words = text.split()  # tokenization
        words = [word for word in words if word not in stop_words]  # removing stop-words
        return " ".join(words)

    df['clean_statement'] = df['statement'].apply(clean_data)
    return df
