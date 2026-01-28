"""
Quick Start Script - For testing with small dataset
Use this for quick testing without processing the full dataset
"""

import sys
sys.path.append('src')

import pandas as pd
import os

# Create a small sample dataset for testing
def create_sample_data():
    """Create a small sample dataset for quick testing"""
    
    sample_tweets = [
        ("I love this product! It's amazing and works perfectly!", 2),
        ("Absolutely wonderful experience. Highly recommend!", 2),
        ("Best purchase I've made in years. So happy!", 2),
        ("Great quality and fast shipping. Very satisfied.", 2),
        ("Fantastic! Exceeded all my expectations!", 2),
        ("This is terrible. Worst product ever.", 0),
        ("Completely disappointed. Waste of money.", 0),
        ("Horrible experience. Never buying again.", 0),
        ("Bad quality and poor customer service.", 0),
        ("Absolutely awful. Total disaster.", 0),
        ("It's okay. Nothing special.", 1),
        ("Average product. Does the job.", 1),
        ("Not bad, not great. Just fine.", 1),
        ("Decent for the price.", 1),
        ("It works as expected.", 1),
    ] * 100  # Repeat to have more samples
    
    df = pd.DataFrame(sample_tweets, columns=['text', 'sentiment'])
    
    # Add cleaned_text (simplified preprocessing)
    df['cleaned_text'] = df['text'].str.lower().str.replace('[^a-zA-Z\s]', '', regex=True)
    df['text_length'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['textblob_sentiment'] = 0.0  # Placeholder
    
    return df


def quick_test():
    """Run a quick test of the sentiment analysis pipeline"""
    
    print("="*70)
    print("              QUICK START - SENTIMENT ANALYSIS TEST")
    print("="*70)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Create sample data
    print("\n1. Creating sample dataset...")
    df = create_sample_data()
    df.to_csv('data/processed/sample_tweets.csv', index=False)
    print(f"✓ Created {len(df)} sample tweets")
    print(f"   - Positive: {len(df[df['sentiment']==2])}")
    print(f"   - Negative: {len(df[df['sentiment']==0])}")
    print(f"   - Neutral: {len(df[df['sentiment']==1])}")
    
    # Train model
    print("\n2. Training sentiment analysis model...")
    from src.sentiment_analysis import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df, test_size=0.2)
    
    analyzer.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate
    print("\n3. Evaluating model...")
    metrics = analyzer.evaluate_model(X_test, y_test)
    analyzer.print_evaluation(metrics, y_test)
    
    # Save model
    analyzer.save_model('models/sample_model.pkl', 'models/sample_vectorizer.pkl')
    
    # Test predictions
    print("\n4. Testing predictions...")
    test_cases = [
        "I absolutely love this! Best ever!",
        "This is horrible and disappointing",
        "It's okay, nothing special"
    ]
    
    for tweet in test_cases:
        _, sentiment = analyzer.predict_sentiment(tweet)
        emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"
        print(f"\nTweet: {tweet}")
        print(f"Sentiment: {sentiment} {emoji}")
    
    # Create basic visualization
    print("\n5. Creating visualizations...")
    from src.visualization import SentimentVisualizer
    
    visualizer = SentimentVisualizer(output_dir='outputs/plots')
    visualizer.plot_sentiment_distribution(df)
    
    print("\n" + "="*70)
    print("                    QUICK TEST COMPLETE!")
    print("="*70)
    print("\n✓ Sample model trained successfully")
    print(f"✓ Accuracy: {metrics['accuracy']:.2%}")
    print("✓ Ready to process full dataset with main.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()