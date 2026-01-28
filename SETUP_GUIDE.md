# Twitter Sentiment Analysis - VS Code Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher installed
- VS Code installed
- Internet connection for downloading dataset

## 🚀 Step-by-Step Setup in VS Code

### Step 1: Create Project Folder

```bash
mkdir twitter-sentiment-analysis
cd twitter-sentiment-analysis
```

### Step 2: Create Project Structure

Create the following folder structure:

```
twitter-sentiment-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── sentiment_analysis.py
│   └── visualization.py
│
├── models/
├── outputs/
│   ├── plots/
│   └── results/
│
├── requirements.txt
├── main.py
├── quick_start.py
└── README.md
```

### Step 3: Create Files in VS Code

1. **Open VS Code** in your project folder:

   ```bash
   code .
   ```

2. **Create `src/__init__.py`** (empty file):
   - Right-click on `src` folder
   - Select "New File"
   - Name it `__init__.py`
   - Leave it empty (just create it)

3. **Copy the code files** I provided:
   - `src/data_preprocessing.py`
   - `src/sentiment_analysis.py`
   - `src/visualization.py`
   - `main.py`
   - `quick_start.py`
   - `requirements.txt`

### Step 4: Create Virtual Environment

1. **Open Terminal in VS Code** (Ctrl + ` or View → Terminal)

2. **Create virtual environment**:

   ```bash
   # Windows
   python -m venv venv

   # Mac/Linux
   python3 -m venv venv
   ```

3. **Activate virtual environment**:

   ```bash
   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

   You should see `(venv)` in your terminal prompt.

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- pandas, numpy, scipy
- nltk, textblob
- scikit-learn
- matplotlib, seaborn, wordcloud
- jupyter (optional)

### Step 6: Download NLTK Data

Run this command in terminal:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 7: Download Dataset

1. **Go to Kaggle**: https://www.kaggle.com/datasets/kazanova/sentiment140

2. **Download the dataset**:
   - You need a Kaggle account (free)
   - Click "Download" button
   - Download `training.1600000.processed.noemoticon.csv`

3. **Place the file**:
   - Move downloaded file to `data/raw/` folder
   - Rename it to `sentiment140.csv`

   Final path: `data/raw/sentiment140.csv`

## ▶️ Running the Project

### Option 1: Quick Test (Recommended for First Run)

Test with sample data first:

```bash
python quick_start.py
```

This will:

- Create sample dataset
- Train a small model
- Test predictions
- Create basic visualization
- Takes ~30 seconds

### Option 2: Full Pipeline

Run the complete analysis:

```bash
python main.py
```

This will:

1. Preprocess tweets (can take 5-10 minutes for 50K tweets)
2. Create visualizations
3. Train models
4. Evaluate performance
5. Interactive prediction mode

### Option 3: Run Individual Components

**Preprocessing only:**

```python
from src.data_preprocessing import preprocess_tweets
df = preprocess_tweets('data/raw/sentiment140.csv', sample_size=10000)
```

**Training only:**

```python
from src.sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
# ... train model
```

**Visualization only:**

```python
from src.visualization import SentimentVisualizer
visualizer = SentimentVisualizer()
# ... create plots
```

## 🎯 VS Code Recommended Extensions

Install these extensions for better experience:

1. **Python** (by Microsoft)
2. **Pylance** (by Microsoft)
3. **Jupyter** (by Microsoft) - if using notebooks
4. **Python Indent** - for better indentation
5. **autoDocstring** - for documentation

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution:**

```bash
# Make sure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "NLTK data not found"

**Solution:**

```python
import nltk
nltk.download('all')  # Downloads all NLTK data
```

### Issue: "Memory Error" during processing

**Solution:**

- Reduce `SAMPLE_SIZE` in `main.py`
- Change from 50000 to 10000 or 5000
- Line 22 in `main.py`: `SAMPLE_SIZE = 10000`

### Issue: "File not found" for dataset

**Solution:**

- Make sure file path is exactly: `data/raw/sentiment140.csv`
- Check file exists: `ls data/raw/` (Mac/Linux) or `dir data\raw\` (Windows)

### Issue: Plots not showing

**Solution:**

```bash
# Install missing backend
pip install PyQt5

# Or use inline plots in Jupyter
%matplotlib inline
```

## 📊 Expected Output

After running `main.py`, you should see:

### Console Output:

```
==========================================================
STEP 1: DATA PREPROCESSING
==========================================================
Loading dataset...
Dataset loaded: 50000 tweets
Preprocessing tweets...
...

Accuracy:  0.7850
Precision: 0.7920
Recall:    0.7810
F1-Score:  0.7860
```

### Generated Files:

```
data/processed/cleaned_tweets.csv
models/sentiment_model.pkl
models/vectorizer.pkl
outputs/plots/sentiment_distribution.png
outputs/plots/wordclouds.png
outputs/plots/confusion_matrix.png
outputs/plots/text_length_distribution.png
outputs/plots/top_words.png
```

## 💡 Tips for VS Code

### Running Python Files:

1. **Method 1**: Right-click file → "Run Python File in Terminal"
2. **Method 2**: Click ▶️ button in top-right
3. **Method 3**: Press `Ctrl+F5`

### Debugging:

1. Set breakpoints by clicking left margin
2. Press `F5` to start debugging
3. Use Debug Console for interactive debugging

### Viewing CSV Files:

- Install "Excel Viewer" extension
- Click CSV file to view as table

### Terminal Shortcuts:

- `Ctrl + `` - Toggle terminal
- `Ctrl + Shift + `` - New terminal
- `Ctrl + C` - Stop running program

## 📈 Customization

### Change Sample Size:

Edit `main.py`, line 22:

```python
SAMPLE_SIZE = 10000  # Smaller = faster, less accurate
```

### Change Model Type:

Edit `main.py`, line 84:

```python
analyzer.train_model(X_train, y_train, model_type='naive_bayes')  # or 'svm'
```

### Adjust Visualizations:

Edit `visualization.py`:

- Change colors
- Adjust plot sizes
- Add new visualizations

## 🎓 Next Steps

1. ✅ Run `quick_start.py` to test setup
2. ✅ Download full dataset
3. ✅ Run `main.py` for complete analysis
4. ✅ Experiment with different models
5. ✅ Try interactive prediction mode
6. ✅ Create custom visualizations
7. ✅ Build a web interface (Streamlit)

## 📞 Need Help?

Common issues:

- Check all files are in correct folders
- Verify virtual environment is activated
- Ensure dataset is downloaded and placed correctly
- Check Python version: `python --version`

Happy analyzing! 🎉
