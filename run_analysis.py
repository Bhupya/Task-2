"""
Interactive Analysis Script
Use this for step-by-step execution (similar to Jupyter notebook)
Run sections separately for better understanding
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_tweets, get_data_statistics, TweetPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.visualization import SentimentVisualizer

# ============================================================================
# SECTION 1: LOAD AND EXPLORE DATA
# ============================================================================

def section_1_load_data():
    """Load and explore the dataset"""
    print("="*70)
    print("SECTION 1: LOADING AND EXPLORING DATA")
    print("="*70)
    
    # Check if preprocessed data exists
    import os
    if os.path.exists("data/processed/cleaned_tweets.csv"):
        print("\nLoading preprocessed data...")
        df = pd.read_csv("data/processed/cleaned_tweets.csv")
    else:
        print("\nPreprocessing raw data...")
        df = preprocess_tweets(
            "data/raw/sentiment140.csv",
            sample_size=10000,  # Start small
            save_path="data/processed/cleaned_tweets.csv"
        )
    
    # Display basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Statistics
    get_data_statistics(df)
    
    return df


# ============================================================================
# SECTION 2: DATA VISUALIZATION
# ============================================================================

def section_2_visualize(df):
    """Create visualizations"""
    print("\n" + "="*70)
    print("SECTION 2: DATA VISUALIZATION")
    print("="*70)
    
    visualizer = SentimentVisualizer()
    
    # Plot sentiment distribution
    print("\nCreating sentiment distribution plot...")
    visualizer.plot_sentiment_distribution(df)
    
    # Word clouds
    print("\nCreating word clouds...")
    visualizer.plot_word_clouds(df)
    
    # Text length distribution
    print("\nCreating text length distribution...")
    visualizer.plot_text_length_distribution(df)
    
    # Top words
    print("\nCreating top words visualization...")
    visualizer.plot_top_words(df, n=15)
    
    print("\n✓ All visualizations created!")
    
    return visualizer


# ============================================================================
# SECTION 3: TEXT PREPROCESSING DETAILS
# ============================================================================

def section_3_preprocessing_demo():
    """Demonstrate text preprocessing steps"""
    print("\n" + "="*70)
    print("SECTION 3: TEXT PREPROCESSING DEMONSTRATION")
    print("="*70)
    
    # Sample tweet
    sample_tweet = "@JohnDoe Check out this amazing product! 😊 https://example.com #awesome"
    
    print(f"\nOriginal Tweet:")
    print(f"  {sample_tweet}")
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor()
    
    # Step by step preprocessing
    print(f"\n1. After cleaning (URLs, mentions, hashtags removed):")
    cleaned = preprocessor.clean_tweet(sample_tweet)
    print(f"  {cleaned}")
    
    print(f"\n2. After removing stopwords:")
    no_stopwords = preprocessor.remove_stopwords(cleaned)
    print(f"  {no_stopwords}")
    
    print(f"\n3. After lemmatization:")
    lemmatized = preprocessor.lemmatize_text(no_stopwords)
    print(f"  {lemmatized}")
    
    # Full pipeline
    print(f"\nFull Pipeline Result:")
    result = preprocessor.preprocess_pipeline(sample_tweet)
    print(f"  {result}")
    
    # TextBlob sentiment
    print(f"\nTextBlob Sentiment Score: {preprocessor.get_sentiment_textblob(sample_tweet):.3f}")
    print("  (Range: -1.0 = very negative, +1.0 = very positive)")


# ============================================================================
# SECTION 4: MODEL TRAINING
# ============================================================================

def section_4_train_model(df):
    """Train sentiment analysis model"""
    print("\n" + "="*70)
    print("SECTION 4: MODEL TRAINING")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
    
    # Prepare data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df, test_size=0.2)
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression model...")
    analyzer.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate
    print("\nEvaluating model performance...")
    metrics = analyzer.evaluate_model(X_test, y_test)
    analyzer.print_evaluation(metrics, y_test)
    
    return analyzer, metrics, X_test, y_test


# ============================================================================
# SECTION 5: MODEL ANALYSIS
# ============================================================================

def section_5_model_analysis(analyzer, metrics, y_test):
    """Analyze model performance"""
    print("\n" + "="*70)
    print("SECTION 5: MODEL ANALYSIS")
    print("="*70)
    
    # Top features
    print("\nTop predictive features for each sentiment:")
    top_features = analyzer.get_top_features(n=10)
    
    if top_features:
        for sentiment, features in top_features.items():
            print(f"\n{sentiment}:")
            for i, feature in enumerate(features, 1):
                print(f"  {i}. {feature}")
    
    # Confusion matrix
    print("\nPlotting confusion matrix...")
    visualizer = SentimentVisualizer()
    visualizer.plot_confusion_matrix(y_test, metrics['predictions'])
    
    # Performance summary
    print("\n" + "-"*70)
    print("PERFORMANCE SUMMARY")
    print("-"*70)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")


# ============================================================================
# SECTION 6: INTERACTIVE PREDICTIONS
# ============================================================================

def section_6_predictions(analyzer):
    """Test model with custom predictions"""
    print("\n" + "="*70)
    print("SECTION 6: INTERACTIVE PREDICTIONS")
    print("="*70)
    
    test_cases = [
        "I absolutely love this product! It's the best thing ever! 😊",
        "This is terrible. Waste of money. Very disappointed.",
        "It's okay. Nothing special, but it works.",
        "Amazing quality! Highly recommend to everyone!",
        "Worst purchase ever. Complete disaster.",
        "Pretty good. Worth the price I paid.",
    ]
    
    print("\nTesting sample tweets:\n")
    
    for i, tweet in enumerate(test_cases, 1):
        sentiment_label, sentiment_name = analyzer.predict_sentiment(tweet)
        
        # Add emoji
        if sentiment_label == 2:
            emoji = "😊"
            color = "Positive"
        elif sentiment_label == 0:
            emoji = "😞"
            color = "Negative"
        else:
            emoji = "😐"
            color = "Neutral"
        
        print(f"{i}. Tweet: {tweet}")
        print(f"   Prediction: {sentiment_name} {emoji}\n")


# ============================================================================
# SECTION 7: MODEL COMPARISON
# ============================================================================

def section_7_compare_models(df):
    """Compare different models"""
    print("\n" + "="*70)
    print("SECTION 7: MODEL COMPARISON")
    print("="*70)
    
    models_to_test = ['logistic', 'naive_bayes', 'svm']
    results = {}
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type.upper()}...")
        
        analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
        X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
        
        analyzer.train_model(X_train, y_train, model_type=model_type)
        metrics = analyzer.evaluate_model(X_test, y_test)
        
        results[model_type] = metrics
    
    # Display comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_sections():
    """Run all sections sequentially"""
    
    print("\n" + "="*70)
    print("    TWITTER SENTIMENT ANALYSIS - INTERACTIVE WALKTHROUGH")
    print("="*70)
    
    # Section 1: Load Data
    df = section_1_load_data()
    input("\nPress Enter to continue to Section 2 (Visualization)...")
    
    # Section 2: Visualize
    visualizer = section_2_visualize(df)
    input("\nPress Enter to continue to Section 3 (Preprocessing Demo)...")
    
    # Section 3: Preprocessing Demo
    section_3_preprocessing_demo()
    input("\nPress Enter to continue to Section 4 (Model Training)...")
    
    # Section 4: Train Model
    analyzer, metrics, X_test, y_test = section_4_train_model(df)
    input("\nPress Enter to continue to Section 5 (Model Analysis)...")
    
    # Section 5: Analyze Model
    section_5_model_analysis(analyzer, metrics, y_test)
    input("\nPress Enter to continue to Section 6 (Predictions)...")
    
    # Section 6: Predictions
    section_6_predictions(analyzer)
    
    # Ask about model comparison
    response = input("\nDo you want to compare different models? (y/n): ")
    if response.lower() == 'y':
        section_7_compare_models(df)
    
    print("\n" + "="*70)
    print("                   ANALYSIS COMPLETE!")
    print("="*70)
    print("\nYou can now:")
    print("  - Check outputs/plots/ for visualizations")
    print("  - Run individual sections by calling their functions")
    print("  - Modify parameters and re-run")
    print("="*70 + "\n")


if __name__ == "__main__":
    # You can run all sections or individual ones
    
    # Option 1: Run everything step by step
    run_all_sections()
    
    # Option 2: Run individual sections (uncomment to use)
    # df = section_1_load_data()
    # section_2_visualize(df)
    # section_3_preprocessing_demo()
    # analyzer, metrics, X_test, y_test = section_4_train_model(df)
    # section_5_model_analysis(analyzer, metrics, y_test)
    # section_6_predictions(analyzer)
    # section_7_compare_models(df)