"""
Visualization Module for Twitter Sentiment Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SentimentVisualizer:
    """Class for creating visualizations"""
    
    def __init__(self, output_dir='outputs/plots'):
        """
        Initialize visualizer
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_sentiment_distribution(self, df, save=True):
        """
        Plot sentiment distribution
        
        Args:
            df (pd.DataFrame): Dataframe with sentiment column
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        
        # Create labels based on actual sentiments present
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_labels = [sentiment_map[idx] for idx in sentiment_counts.index]
        
        # Assign colors based on sentiment type
        color_map = {0: '#FF6B6B', 1: '#FFD93D', 2: '#6BCB77'}
        colors = [color_map[idx] for idx in sentiment_counts.index]
        
        bars = plt.bar(range(len(sentiment_counts)), sentiment_counts.values, color=colors, alpha=0.8)
        
        plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Tweets', fontsize=12, fontweight='bold')
        plt.title('Sentiment Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xticks(range(len(sentiment_counts)), sentiment_labels, fontsize=11)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Sentiment distribution plot saved to {self.output_dir}/sentiment_distribution.png")
        
        plt.close()  # Close instead of show to avoid blocking
    
    def plot_word_clouds(self, df, save=True):
        """
        Create word clouds for each sentiment
        
        Args:
            df (pd.DataFrame): Dataframe with cleaned_text and sentiment
            save (bool): Whether to save the plots
        """
        sentiments = {
            0: ('Negative', '#FF6B6B'),
            2: ('Positive', '#6BCB77')
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (sentiment_val, (sentiment_name, color)) in enumerate(sentiments.items()):
            text = ' '.join(df[df['sentiment'] == sentiment_val]['cleaned_text'].values)
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='RdYlGn' if sentiment_val == 2 else 'Reds',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(text)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'{sentiment_name} Tweets Word Cloud', 
                               fontsize=14, fontweight='bold', pad=20)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/wordclouds.png', dpi=300, bbox_inches='tight')
            print(f"✓ Word clouds saved to {self.output_dir}/wordclouds.png")
        
        plt.close()  # Close instead of show
    
    def plot_confusion_matrix(self, y_test, y_pred, save=True):
        """
        Plot confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Get unique labels from the data
        unique_labels = sorted(set(y_test) | set(y_pred))
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        label_names = [sentiment_map[label] for label in unique_labels]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names,
                   yticklabels=label_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {self.output_dir}/confusion_matrix.png")
        
        plt.close()  # Close instead of show
    
    def plot_text_length_distribution(self, df, save=True):
        """
        Plot distribution of tweet lengths
        
        Args:
            df (pd.DataFrame): Dataframe with text_length column
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        color_map = {0: '#FF6B6B', 1: '#FFD93D', 2: '#6BCB77'}
        
        # Get unique sentiments in the data
        unique_sentiments = sorted(df['sentiment'].unique())
        
        for sentiment_val in unique_sentiments:
            sentiment_name = sentiment_map[sentiment_val]
            color = color_map[sentiment_val]
            data = df[df['sentiment'] == sentiment_val]['text_length']
            plt.hist(data, bins=50, alpha=0.6, label=sentiment_name, 
                    color=color, edgecolor='black')
        
        plt.xlabel('Tweet Length (characters)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Tweet Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/text_length_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Text length distribution saved to {self.output_dir}/text_length_distribution.png")
        
        plt.close()  # Close instead of show
    
    def plot_word_count_distribution(self, df, save=True):
        """
        Plot distribution of word counts
        
        Args:
            df (pd.DataFrame): Dataframe with word_count column
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        color_map = {0: '#FF6B6B', 1: '#FFD93D', 2: '#6BCB77'}
        
        # Get unique sentiments in the data
        unique_sentiments = sorted(df['sentiment'].unique())
        
        for sentiment_val in unique_sentiments:
            sentiment_name = sentiment_map[sentiment_val]
            color = color_map[sentiment_val]
            data = df[df['sentiment'] == sentiment_val]['word_count']
            plt.hist(data, bins=30, alpha=0.6, label=sentiment_name, 
                    color=color, edgecolor='black')
        
        plt.xlabel('Word Count', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/word_count_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Word count distribution saved to {self.output_dir}/word_count_distribution.png")
        
        plt.close()  # Close instead of show
    
    def plot_sentiment_comparison(self, df, save=True):
        """
        Compare actual sentiment vs TextBlob sentiment
        
        Args:
            df (pd.DataFrame): Dataframe with sentiment and textblob_sentiment
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        color_map = {0: '#FF6B6B', 1: '#FFD93D', 2: '#6BCB77'}
        
        # Get unique sentiments in the data
        unique_sentiments = sorted(df['sentiment'].unique())
        
        for sentiment_val in unique_sentiments:
            sentiment_name = sentiment_map[sentiment_val]
            color = color_map[sentiment_val]
            data = df[df['sentiment'] == sentiment_val]['textblob_sentiment']
            plt.scatter(data.index[:1000], data[:1000], 
                      alpha=0.5, label=sentiment_name, 
                      color=color, s=10)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.xlabel('Tweet Index', fontsize=12, fontweight='bold')
        plt.ylabel('TextBlob Sentiment Polarity', fontsize=12, fontweight='bold')
        plt.title('TextBlob Sentiment Polarity by Actual Sentiment', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/sentiment_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Sentiment comparison saved to {self.output_dir}/sentiment_comparison.png")
        
        plt.close()  # Close instead of show
    
    def plot_top_words(self, df, n=15, save=True):
        """
        Plot top words for each sentiment
        
        Args:
            df (pd.DataFrame): Dataframe with cleaned_text and sentiment
            n (int): Number of top words to show
            save (bool): Whether to save the plot
        """
        from collections import Counter
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sentiments = [(0, 'Negative', '#FF6B6B'), (2, 'Positive', '#6BCB77')]
        
        for idx, (sentiment_val, sentiment_name, color) in enumerate(sentiments):
            text = ' '.join(df[df['sentiment'] == sentiment_val]['cleaned_text'].values)
            words = text.split()
            word_counts = Counter(words).most_common(n)
            
            words_list = [word for word, count in word_counts]
            counts_list = [count for word, count in word_counts]
            
            axes[idx].barh(words_list, counts_list, color=color, alpha=0.8)
            axes[idx].set_xlabel('Frequency', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'Top {n} Words in {sentiment_name} Tweets', 
                              fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/top_words.png', dpi=300, bbox_inches='tight')
            print(f"✓ Top words plot saved to {self.output_dir}/top_words.png")
        
        plt.close()  # Close instead of show
    
    def create_all_visualizations(self, df, y_test=None, y_pred=None):
        """
        Create all visualizations
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            y_test: True test labels (optional)
            y_pred: Predicted labels (optional)
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_sentiment_distribution(df)
        self.plot_word_clouds(df)
        self.plot_text_length_distribution(df)
        self.plot_word_count_distribution(df)
        self.plot_top_words(df)
        
        if 'textblob_sentiment' in df.columns:
            self.plot_sentiment_comparison(df)
        
        if y_test is not None and y_pred is not None:
            self.plot_confusion_matrix(y_test, y_pred)
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS CREATED!")
        print("="*60)


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/cleaned_tweets.csv")
    
    # Create visualizer
    visualizer = SentimentVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations(df)