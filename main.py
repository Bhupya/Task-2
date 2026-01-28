"""
Main Script for Twitter Sentiment Analysis
Run this file to execute the complete pipeline
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.data_preprocessing import preprocess_tweets, get_data_statistics
from src.sentiment_analysis import SentimentAnalyzer, compare_models
from src.visualization import SentimentVisualizer
import pandas as pd


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'outputs/plots',
        'outputs/results'
    ]
    
    for directory in directories:
        try:
            # Remove if it's a file instead of directory
            if os.path.exists(directory) and os.path.isfile(directory):
                os.remove(directory)
                print(f"⚠ Removed conflicting file: {directory}")
            
            # Create directory
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"⚠ Warning creating {directory}: {e}")
            # Try to continue anyway
            pass
    
    print("✓ Directory structure verified")


def main():
    """Main execution function"""
    
    print("="*70)
    print("        TWITTER SENTIMENT ANALYSIS - COMPLETE PIPELINE")
    print("="*70)
    
    # Setup directories
    try:
        setup_directories()
    except Exception as e:
        print(f"Warning during directory setup: {e}")
        print("Attempting to continue...")
    
    # Configuration
    RAW_DATA_PATH = "data/raw/sentiment140.csv"
    PROCESSED_DATA_PATH = "data/processed/cleaned_tweets.csv"
    MODEL_PATH = "models/sentiment_model.pkl"
    VECTORIZER_PATH = "models/vectorizer.pkl"
    SAMPLE_SIZE = 50000  # Adjust based on your system (None for full dataset)
    
    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n❌ Error: Dataset not found at {RAW_DATA_PATH}")
        print("\n📥 Please download the Sentiment140 dataset from:")
        print("   https://www.kaggle.com/datasets/kazanova/sentiment140")
        print(f"\n📁 Place the file at: {RAW_DATA_PATH}")
        print("\n💡 OR run 'python quick_start.py' to test with sample data")
        return
    
    # Step 1: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"\n🔄 Processing {SAMPLE_SIZE if SAMPLE_SIZE else 'all'} tweets from raw data...")
        df = preprocess_tweets(
            RAW_DATA_PATH, 
            sample_size=SAMPLE_SIZE,
            save_path=PROCESSED_DATA_PATH
        )
        get_data_statistics(df)
    else:
        print(f"\n✓ Loading existing preprocessed data from {PROCESSED_DATA_PATH}...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"✓ Loaded {len(df)} preprocessed tweets")
        get_data_statistics(df)
    
    # Step 2: Visualization
    print("\n" + "="*70)
    print("STEP 2: DATA VISUALIZATION")
    print("="*70)
    
    visualizer = SentimentVisualizer()
    
    print("\n📊 Creating visualizations...")
    print("   1. Sentiment distribution chart")
    visualizer.plot_sentiment_distribution(df)
    
    print("   2. Word clouds for positive/negative tweets")
    visualizer.plot_word_clouds(df)
    
    print("   3. Text length distribution")
    visualizer.plot_text_length_distribution(df)
    
    print("   4. Top words analysis")
    visualizer.plot_top_words(df, n=15)
    
    print("\n✓ All visualizations saved to 'outputs/plots/'")
    
    # Step 3: Model Training
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(vectorizer_type='tfidf')
    
    # Prepare data
    print("\n📂 Preparing training and test data...")
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df, test_size=0.2)
    
    # Train model (Logistic Regression - best performer)
    print("\n🤖 Training Logistic Regression model...")
    analyzer.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    metrics = analyzer.evaluate_model(X_test, y_test)
    analyzer.print_evaluation(metrics, y_test)
    
    # Plot confusion matrix
    print("\n📊 Creating confusion matrix...")
    visualizer.plot_confusion_matrix(y_test, metrics['predictions'])
    
    # Show top features
    print("\n" + "="*70)
    print("TOP PREDICTIVE FEATURES")
    print("="*70)
    
    top_features = analyzer.get_top_features(n=10)
    if top_features:
        for sentiment, features in top_features.items():
            print(f"\n✨ {sentiment} Sentiment Keywords:")
            print(f"   {', '.join(features)}")
    
    # Save model
    print("\n💾 Saving trained model...")
    analyzer.save_model(MODEL_PATH, VECTORIZER_PATH)
    
    # Step 4: Model Comparison (Optional)
    print("\n" + "="*70)
    print("STEP 4: MODEL COMPARISON (OPTIONAL)")
    print("="*70)
    
    user_input = input("\n❓ Do you want to compare different models (Logistic, Naive Bayes, SVM)? (y/n): ")
    if user_input.lower() == 'y':
        print("\n🔄 Comparing models (this may take a few minutes)...")
        compare_models(df)
    else:
        print("⏭️  Skipping model comparison...")
    
    # Step 5: Interactive Prediction
    print("\n" + "="*70)
    print("STEP 5: INTERACTIVE SENTIMENT PREDICTION")
    print("="*70)
    
    # Test some sample tweets
    test_tweets = [
        "I absolutely love this! Best day ever! 😊",
        "This is terrible. Worst experience of my life.",
        "It's okay, nothing special really.",
        "Amazing product! Highly recommend to everyone!",
        "Disappointed with the quality. Would not buy again."
    ]
    
    print("\n🧪 Testing sample tweets:\n")
    for i, tweet in enumerate(test_tweets, 1):
        sentiment_label, sentiment_name = analyzer.predict_sentiment(tweet)
        emoji = "😊" if sentiment_label == 2 else "😞" if sentiment_label == 0 else "😐"
        print(f"{i}. Tweet: {tweet}")
        print(f"   Prediction: {sentiment_name} {emoji}\n")
    
    # Interactive mode
    print("\n" + "-"*70)
    print("🎮 INTERACTIVE MODE - Test your own tweets!")
    print("-"*70)
    print("💡 Type any tweet to analyze its sentiment")
    print("💡 Type 'quit', 'exit', or 'q' to finish\n")
    
    while True:
        user_tweet = input("✍️  Enter a tweet to analyze: ")
        
        if user_tweet.lower() in ['quit', 'exit', 'q', '']:
            break
        
        if user_tweet.strip():
            sentiment_label, sentiment_name = analyzer.predict_sentiment(user_tweet)
            emoji = "😊" if sentiment_label == 2 else "😞" if sentiment_label == 0 else "😐"
            
            # Add color coding in output
            if sentiment_name == "Positive":
                symbol = "✅"
            elif sentiment_name == "Negative":
                symbol = "❌"
            else:
                symbol = "➖"
            
            print(f"   {symbol} Predicted Sentiment: {sentiment_name} {emoji}\n")
    
    # Summary
    print("\n" + "="*70)
    print("                     🎉 PIPELINE COMPLETE! 🎉")
    print("="*70)
    print("\n✅ Summary of Results:")
    print(f"   • Data preprocessed: {len(df):,} tweets")
    print(f"   • Model accuracy: {metrics['accuracy']:.2%}")
    print(f"   • Model F1-score: {metrics['f1_score']:.2%}")
    print(f"   • Precision: {metrics['precision']:.2%}")
    print(f"   • Recall: {metrics['recall']:.2%}")
    
    print("\n📁 Output Files Generated:")
    print(f"   • Processed data: {PROCESSED_DATA_PATH}")
    print(f"   • Trained model: {MODEL_PATH}")
    print(f"   • Visualizations: outputs/plots/ (6 PNG files)")
    
    print("\n🚀 Next Steps:")
    print("   • View visualizations in 'outputs/plots/' folder")
    print("   • Check processed data in 'data/processed/'")
    print("   • Use saved model for future predictions")
    print("   • Run 'python run_analysis.py' for detailed walkthrough")
    
    print("\n" + "="*70)
    print("         Thank you for using Twitter Sentiment Analysis!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user (Ctrl+C)")
        print("💡 You can resume by running 'python main.py' again")
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {str(e)}")
        print("💡 Make sure the dataset is at: data/raw/sentiment140.csv")
        print("💡 Or run 'python quick_start.py' to test with sample data")
    except ImportError as e:
        print(f"\n❌ Import error: {str(e)}")
        print("💡 Make sure you've installed all requirements:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\n📋 Full error details:")
        import traceback
        traceback.print_exc()
        print("\n💡 Tips:")
        print("   • Check if virtual environment is activated")
        print("   • Verify all dependencies are installed")
        print("   • Make sure dataset exists at correct location")
        print("   • Try running 'python quick_start.py' first")