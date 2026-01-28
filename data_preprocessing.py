"""
Data Preprocessing Module for Twitter Sentiment Analysis
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TweetPreprocessor:
    """Class to handle tweet preprocessing operations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_tweet(self, text):
        """
        Clean individual tweet text
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned tweet text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Cleaned tweet text
            
        Returns:
            str: Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text to reduce words to base form
        
        Args:
            text (str): Text to lemmatize
            
        Returns:
            str: Lemmatized text
        """
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def get_sentiment_textblob(self, text):
        """
        Get sentiment polarity using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment polarity (-1 to 1)
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def preprocess_pipeline(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Fully preprocessed text
        """
        text = self.clean_tweet(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text


def load_sentiment140_data(filepath, sample_size=None):
    """
    Load Sentiment140 dataset
    
    Args:
        filepath (str): Path to sentiment140.csv
        sample_size (int): Number of samples to load (None for all)
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Column names for Sentiment140 dataset
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    print("Loading dataset...")
    df = pd.read_csv(filepath, encoding='latin-1', names=columns)
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset loaded: {len(df)} tweets")
    return df


def preprocess_tweets(filepath, sample_size=None, save_path=None):
    """
    Main function to preprocess tweets
    
    Args:
        filepath (str): Path to raw data
        sample_size (int): Number of samples to process
        save_path (str): Path to save processed data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Load data
    df = load_sentiment140_data(filepath, sample_size)
    
    # Convert sentiment labels (0=negative, 4=positive) to (0=negative, 1=neutral, 2=positive)
    df['sentiment'] = df['target'].map({0: 0, 2: 1, 4: 2})
    
    # Keep only necessary columns
    df = df[['text', 'sentiment']]
    
    # Remove missing values
    df = df.dropna()
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor()
    
    # Apply preprocessing
    print("Preprocessing tweets...")
    df['cleaned_text'] = df['text'].apply(preprocessor.preprocess_pipeline)
    
    # Remove empty tweets after preprocessing
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Add text length feature
    df['text_length'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # Add TextBlob sentiment for comparison
    print("Calculating TextBlob sentiment...")
    df['textblob_sentiment'] = df['text'].apply(preprocessor.get_sentiment_textblob)
    
    print(f"Preprocessing complete: {len(df)} tweets processed")
    
    # Save processed data
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")
    
    return df


def get_data_statistics(df):
    """
    Print dataset statistics
    
    Args:
        df (pd.DataFrame): Dataframe to analyze
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal tweets: {len(df)}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts().sort_index())
    
    print(f"\nAverage text length: {df['text_length'].mean():.2f} characters")
    print(f"Average word count: {df['word_count'].mean():.2f} words")
    
    print(f"\nSample tweets:")
    for i in range(3):
        print(f"\n{i+1}. Original: {df.iloc[i]['text'][:100]}...")
        print(f"   Cleaned: {df.iloc[i]['cleaned_text'][:100]}...")
        print(f"   Sentiment: {df.iloc[i]['sentiment']}")


if __name__ == "__main__":
    # Test preprocessing
    filepath = "data/raw/sentiment140.csv"
    df = preprocess_tweets(filepath, sample_size=10000, save_path="data/processed/cleaned_tweets.csv")
    get_data_statistics(df)