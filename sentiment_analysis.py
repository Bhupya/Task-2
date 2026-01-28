"""
Sentiment Analysis Model Training and Evaluation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Class for sentiment analysis model training and prediction"""
    
    def __init__(self, vectorizer_type='tfidf'):
        """
        Initialize sentiment analyzer
        
        Args:
            vectorizer_type (str): 'tfidf' or 'count'
        """
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2
            )
        
        self.model = None
        self.model_name = None
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare train-test split
        
        Args:
            df (pd.DataFrame): Dataframe with cleaned_text and sentiment
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df['cleaned_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='logistic'):
        """
        Train sentiment classification model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            model_type (str): 'logistic', 'naive_bayes', or 'svm'
            
        Returns:
            model: Trained model
        """
        print(f"\nTraining {model_type} model...")
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Select and train model
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model_name = "Logistic Regression"
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
            self.model_name = "Multinomial Naive Bayes"
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000, random_state=42)
            self.model_name = "Linear SVM"
        else:
            raise ValueError("Invalid model_type. Choose 'logistic', 'naive_bayes', or 'svm'")
        
        self.model.fit(X_train_vec, y_train)
        print(f"{self.model_name} training complete!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            dict: Dictionary of metrics
        """
        # Vectorize test data
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def print_evaluation(self, metrics, y_test):
        """
        Print evaluation metrics
        
        Args:
            metrics (dict): Metrics dictionary
            y_test: True labels
        """
        print("\n" + "="*60)
        print(f"MODEL EVALUATION: {self.model_name}")
        print("="*60)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        
        # Get unique labels and create appropriate target names
        unique_labels = sorted(set(y_test) | set(metrics['predictions']))
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        target_names = [sentiment_map[label] for label in unique_labels]
        
        print(classification_report(
            y_test, 
            metrics['predictions'],
            target_names=target_names,
            labels=unique_labels
        ))
        
        print("\n" + "-"*60)
        print("Confusion Matrix:")
        print("-"*60)
        print(metrics['confusion_matrix'])
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for new text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (sentiment_label, sentiment_name)
        """
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        
        # Map to sentiment name
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_name = sentiment_map[prediction]
        
        return prediction, sentiment_name
    
    def get_top_features(self, n=20):
        """
        Get top features for each class
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            dict: Top features for each sentiment
        """
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            
            top_features = {}
            sentiment_names = ['Negative', 'Neutral', 'Positive']
            
            for i, sentiment in enumerate(sentiment_names):
                if i < coef.shape[0]:
                    top_indices = np.argsort(coef[i])[-n:][::-1]
                    top_features[sentiment] = [feature_names[idx] for idx in top_indices]
            
            return top_features
        else:
            return None
    
    def save_model(self, model_path, vectorizer_path):
        """
        Save trained model and vectorizer
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str): Path to save vectorizer
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load trained model and vectorizer
        
        Args:
            model_path (str): Path to model
            vectorizer_path (str): Path to vectorizer
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"Model loaded from {model_path}")


def compare_models(df):
    """
    Compare different models
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
    """
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    results = {}
    model_types = ['logistic', 'naive_bayes', 'svm']
    
    for model_type in model_types:
        analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
        X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
        
        analyzer.train_model(X_train, y_train, model_type=model_type)
        metrics = analyzer.evaluate_model(X_test, y_test)
        
        results[model_type] = {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        }
    
    print("\n" + "-"*60)
    print("Model Comparison Results:")
    print("-"*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for model, scores in results.items():
        print(f"{model:<25} {scores['accuracy']:<12.4f} {scores['f1_score']:<12.4f}")


if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("data/processed/cleaned_tweets.csv")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    
    # Train model
    analyzer.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate model
    metrics = analyzer.evaluate_model(X_test, y_test)
    analyzer.print_evaluation(metrics, y_test)
    
    # Show top features
    top_features = analyzer.get_top_features(n=10)
    if top_features:
        print("\n" + "-"*60)
        print("Top Features for Each Sentiment:")
        print("-"*60)
        for sentiment, features in top_features.items():
            print(f"\n{sentiment}: {', '.join(features)}")
    
    # Save model
    analyzer.save_model('models/sentiment_model.pkl', 'models/vectorizer.pkl')
    
    # Test prediction
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_tweets = [
        "I love this product! It's absolutely amazing!",
        "This is the worst experience ever. Terrible service.",
        "It's okay, nothing special."
    ]
    
    for tweet in test_tweets:
        sentiment_label, sentiment_name = analyzer.predict_sentiment(tweet)
        print(f"\nTweet: {tweet}")
        print(f"Predicted Sentiment: {sentiment_name}")