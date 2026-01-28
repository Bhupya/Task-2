"""
Save detailed analysis results to text files
"""

import sys
sys.path.append('src')

import pandas as pd
import os
from datetime import datetime

def save_results():
    """Save all analysis results to text files"""
    
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("📊 Loading analysis data...")
    df = pd.read_csv("data/processed/cleaned_tweets.csv")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ============================================================
    # 1. DATASET STATISTICS
    # ============================================================
    stats_file = f"{results_dir}/dataset_statistics_{timestamp}.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TWITTER SENTIMENT ANALYSIS - DATASET STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Tweets: {len(df):,}\n")
        f.write(f"Number of Features: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")
        
        f.write("SENTIMENT DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        for sentiment_val, count in sentiment_counts.items():
            sentiment_name = sentiment_map[sentiment_val]
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment_name}: {count:,} ({percentage:.2f}%)\n")
        
        f.write("\n")
        f.write("TEXT STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Average Text Length: {df['text_length'].mean():.2f} characters\n")
        f.write(f"Min Text Length: {df['text_length'].min()} characters\n")
        f.write(f"Max Text Length: {df['text_length'].max()} characters\n")
        f.write(f"Median Text Length: {df['text_length'].median():.2f} characters\n\n")
        
        f.write(f"Average Word Count: {df['word_count'].mean():.2f} words\n")
        f.write(f"Min Word Count: {df['word_count'].min()} words\n")
        f.write(f"Max Word Count: {df['word_count'].max()} words\n")
        f.write(f"Median Word Count: {df['word_count'].median():.2f} words\n\n")
        
        f.write("SAMPLE TWEETS\n")
        f.write("-"*70 + "\n")
        for i in range(min(10, len(df))):
            f.write(f"\n{i+1}. Original: {df.iloc[i]['text'][:100]}...\n")
            f.write(f"   Cleaned: {df.iloc[i]['cleaned_text'][:100]}...\n")
            f.write(f"   Sentiment: {sentiment_map[df.iloc[i]['sentiment']]}\n")
            f.write(f"   Length: {df.iloc[i]['text_length']} chars, {df.iloc[i]['word_count']} words\n")
    
    print(f"✓ Dataset statistics saved to: {stats_file}")
    
    # ============================================================
    # 2. TOP WORDS ANALYSIS
    # ============================================================
    words_file = f"{results_dir}/top_words_analysis_{timestamp}.txt"
    
    from collections import Counter
    
    with open(words_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TOP WORDS ANALYSIS BY SENTIMENT\n")
        f.write("="*70 + "\n\n")
        
        for sentiment_val in sorted(df['sentiment'].unique()):
            sentiment_name = sentiment_map[sentiment_val]
            f.write(f"\n{sentiment_name.upper()} SENTIMENT - TOP 30 WORDS\n")
            f.write("-"*70 + "\n")
            
            # Get all text for this sentiment
            text = ' '.join(df[df['sentiment'] == sentiment_val]['cleaned_text'].values)
            words = text.split()
            word_counts = Counter(words).most_common(30)
            
            for rank, (word, count) in enumerate(word_counts, 1):
                f.write(f"{rank:2d}. {word:<20s} - {count:,} occurrences\n")
    
    print(f"✓ Top words analysis saved to: {words_file}")
    
    # ============================================================
    # 3. MODEL PERFORMANCE (if exists)
    # ============================================================
    try:
        import pickle
        
        # Check if model exists
        if os.path.exists('models/sentiment_model.pkl'):
            from src.sentiment_analysis import SentimentAnalyzer
            
            # Load model
            analyzer = SentimentAnalyzer()
            analyzer.load_model('models/sentiment_model.pkl', 'models/vectorizer.pkl')
            
            # Prepare test data
            X_train, X_test, y_train, y_test = analyzer.prepare_data(df, test_size=0.2)
            
            # Evaluate
            metrics = analyzer.evaluate_model(X_test, y_test)
            
            # Save results
            model_file = f"{results_dir}/model_performance_{timestamp}.txt"
            
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("MODEL PERFORMANCE REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Model: {analyzer.model_name}\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-"*70 + "\n")
                f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
                
                f.write("CONFUSION MATRIX\n")
                f.write("-"*70 + "\n")
                f.write(str(metrics['confusion_matrix']))
                f.write("\n\n")
                
                # Top features
                top_features = analyzer.get_top_features(n=20)
                if top_features:
                    f.write("TOP PREDICTIVE FEATURES\n")
                    f.write("-"*70 + "\n\n")
                    for sentiment, features in top_features.items():
                        f.write(f"{sentiment} Sentiment Keywords:\n")
                        for i, feature in enumerate(features, 1):
                            f.write(f"  {i:2d}. {feature}\n")
                        f.write("\n")
            
            print(f"✓ Model performance saved to: {model_file}")
    
    except Exception as e:
        print(f"⚠ Could not save model results: {e}")
        print("  (Model may not be trained yet - run 'python main.py' first)")
    
    # ============================================================
    # 4. SUMMARY REPORT
    # ============================================================
    summary_file = f"{results_dir}/summary_report_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TWITTER SENTIMENT ANALYSIS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"1. Dataset Size: {len(df):,} tweets analyzed\n\n")
        
        f.write("2. Sentiment Breakdown:\n")
        for sentiment_val, count in sentiment_counts.items():
            sentiment_name = sentiment_map[sentiment_val]
            percentage = (count / len(df)) * 100
            f.write(f"   - {sentiment_name}: {percentage:.1f}%\n")
        
        f.write(f"\n3. Average tweet length: {df['text_length'].mean():.0f} characters\n")
        f.write(f"4. Average word count: {df['word_count'].mean():.1f} words per tweet\n\n")
        
        if os.path.exists('models/sentiment_model.pkl'):
            f.write(f"5. Model Accuracy: {metrics['accuracy']*100:.1f}%\n")
            f.write(f"6. Model F1-Score: {metrics['f1_score']*100:.1f}%\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-"*70 + "\n")
        f.write("Visualizations:\n")
        if os.path.exists('outputs/plots'):
            plots = [f for f in os.listdir('outputs/plots') if f.endswith('.png')]
            for plot in plots:
                f.write(f"  ✓ {plot}\n")
        
        f.write("\nData Files:\n")
        f.write(f"  ✓ Processed tweets: data/processed/cleaned_tweets.csv\n")
        
        if os.path.exists('models/sentiment_model.pkl'):
            f.write("\nModel Files:\n")
            f.write(f"  ✓ Trained model: models/sentiment_model.pkl\n")
            f.write(f"  ✓ Vectorizer: models/vectorizer.pkl\n")
        
        f.write("\nResult Files:\n")
        f.write(f"  ✓ Dataset statistics: {os.path.basename(stats_file)}\n")
        f.write(f"  ✓ Top words analysis: {os.path.basename(words_file)}\n")
        if os.path.exists('models/sentiment_model.pkl'):
            f.write(f"  ✓ Model performance: {os.path.basename(model_file)}\n")
        f.write(f"  ✓ Summary report: {os.path.basename(summary_file)}\n")
    
    print(f"✓ Summary report saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("ALL RESULTS SAVED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults location: {os.path.abspath(results_dir)}")
    print("\nGenerated files:")
    print(f"  1. {os.path.basename(stats_file)}")
    print(f"  2. {os.path.basename(words_file)}")
    if os.path.exists('models/sentiment_model.pkl'):
        print(f"  3. {os.path.basename(model_file)}")
    print(f"  4. {os.path.basename(summary_file)}")
    print("\n✓ You can now open these .txt files to view detailed results!")

if __name__ == "__main__":
    save_results()