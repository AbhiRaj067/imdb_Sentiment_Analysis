# 🎬 IMDb Movie Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-88.5%25-success.svg)

## 📌 Overview

This project implements **binary sentiment classification** on 50,000 IMDb movie reviews using Natural Language Processing (NLP) and Machine Learning. The system automatically determines whether a movie review expresses positive or negative sentiment.

**Two notebook versions provided:**
- **Full Version** (`imdb_sentiment_analysis_full.ipynb`) - Complete analysis with EDA, visualizations, and model comparison
- **Simple Version** (`imdb_sentiment_analysis_simple.ipynb`) - Streamlined implementation with best model only

**Key Achievement**: 88.5% accuracy using Logistic Regression with TF-IDF vectorization

---

## 🎯 Purpose & Problem Statement

### The Problem

Movie studios, streaming platforms, and review aggregators receive thousands of user reviews daily. Manually reading and categorizing these reviews is:

- ⏰ **Time-consuming** and costly
- 📊 **Difficult to scale**
- 🔍 **Prone to human bias** and inconsistency

### The Solution

An automated sentiment classification system that:

- ✅ Instantly analyzes review sentiment (positive/negative)
- ✅ Processes thousands of reviews in minutes
- ✅ Provides confidence scores for each prediction
- ✅ Enables data-driven decision making

### Real-World Applications

- **Content Platforms**: Quickly gauge audience reaction to new releases
- **Marketing Teams**: Identify trending positive/negative sentiment
- **Product Managers**: Understand what audiences like or dislike
- **Review Aggregators**: Automate sentiment tagging

---

## 📊 Dataset Details

**Source**: [Stanford Large Movie Review Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/)

### Dataset Characteristics

| Attribute | Details |
|-----------|---------|
| **Total Reviews** | 50,000 labeled movie reviews from IMDb |
| **Training Set** | 25,000 reviews (12,500 positive + 12,500 negative) |
| **Test Set** | 25,000 reviews (12,500 positive + 12,500 negative) |
| **Balance** | Perfectly balanced (50-50 split) |
| **Format** | Raw text files with HTML tags, punctuation, mixed case |
| **Avg Review Length** | ~230 words |
| **Language** | English |
| **Download Size** | ~80 MB (compressed) |

### Data Challenges

- 🏷️ **HTML tags** (`<br />`, `<p>`, etc.) need cleaning
- 🔤 **Mixed case** text requires normalization
- 🔢 **Numbers and special characters** add noise
- 📝 **Stopwords** ("the", "is", "a") don't carry sentiment
- 🎭 **Sarcasm** and nuanced language patterns

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.8+ |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ |
| **NLP Libraries** | NLTK 3.8+ (stopwords, lemmatization), BeautifulSoup4 4.12+, Regex |
| **Machine Learning** | Scikit-learn 1.3+ |
| **Text Vectorization** | CountVectorizer (Bag-of-Words), TfidfVectorizer (TF-IDF) |
| **ML Models** | Multinomial Naive Bayes, Logistic Regression |
| **Visualization** | Matplotlib 3.7+, Seaborn 0.12+, WordCloud 1.9+ |
| **Model Serialization** | Joblib 1.3+ |
| **Development Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---


## 🔄 Complete Project Pipeline

### Phase 1: Data Acquisition

The first phase downloads the aclImdb_v1.tar.gz file (80MB) from Stanford, extracts the tar.gz archive to a local directory, parses 50,000 text files from the pos and neg folders, converts data to structured Pandas DataFrames, and saves the results as train.csv and test.csv for faster future loading.

### Phase 2: Text Preprocessing

Text preprocessing takes raw text input like "This movie was <br />TERRIBLE! I hated it. 10/10 would not recommend." and applies step-by-step cleaning. First, HTML tags are removed using BeautifulSoup parser to extract plain text, resulting in "This movie was TERRIBLE! I hated it. 10/10 would not recommend." Next, non-letters are removed using regex filtering to keep only alphabetic characters, producing "This movie was TERRIBLE I hated it would not recommend". Then lowercasing normalizes the text to "this movie was terrible i hated it would not recommend". Tokenization splits this into individual words. Stopword removal using NLTK's English stopword list removes common words like "this", "was", "i", "it", leaving meaningful words like "movie", "terrible", "hated", "would", "recommend". Finally, lemmatization converts words to their base form, so "hated" becomes "hate", producing the final cleaned output: "movie terrible hate would recommend". The preprocessing techniques used are HTML removal with BeautifulSoup parser that extracts plain text, regex filtering that keeps only alphabetic characters (a-z, A-Z), lowercasing to normalize text to lowercase, stopword removal using NLTK's English stopword list, and lemmatization using WordNetLemmatizer to convert words to base form such as "movies" to "movie" and "loved" to "love".

### Phase 3: Exploratory Data Analysis

Exploratory Data Analysis is available only in the full version and includes creating a class distribution plot to visualize positive versus negative review counts showing the balanced 50-50 split, generating word clouds for both positive reviews to show most common words in positive sentiment and negative reviews to show most common words in negative sentiment, and calculating text statistics including average review length, vocabulary size, and most frequent words per class.

### Phase 4: Feature Engineering

Feature engineering implements two approaches. Approach 1 is Bag-of-Words using CountVectorizer, which builds a vocabulary from training data and converts each review to a word count vector, so for example "great movie" becomes a sparse vector. Approach 2 is TF-IDF using TfidfVectorizer, which calculates Term Frequency (TF) to measure how often a word appears in a document, calculates Inverse Document Frequency (IDF) to measure how rare a word is across all documents, and computes TF-IDF as TF multiplied by IDF where higher weight means the word is important and discriminative, so common words like "the" get low weight while specific words like "masterpiece" get high weight.

### Phase 5: Model Training

Model training implements two models. Model 1 is Naive Bayes with Bag-of-Words, using a pipeline of CountVectorizer followed by MultinomialNB. This model learns the probability of each word appearing in positive versus negative reviews, uses Bayes' theorem, and provides fast training with good baseline performance. Model 2 is Logistic Regression with TF-IDF, using a pipeline of TfidfVectorizer followed by LogisticRegression with max_iter set to 1000. This model learns a weight for each TF-IDF feature where positive weights push toward the positive class and negative weights push toward the negative class, using a linear decision boundary with interpretable coefficients.

### Phase 6: Model Evaluation

Model evaluation uses several metrics. Accuracy measures overall correct predictions divided by total predictions. Precision measures true positives divided by the sum of true positives and false positives. Recall measures true positives divided by the sum of true positives and false negatives. F1-Score is the harmonic mean of precision and recall. The confusion matrix provides a visual breakdown of true positives, true negatives, false positives, and false negatives.

### Phase 7: Model Persistence

Model persistence is available in the full version and saves trained models to disk using joblib, creating nb_model.pkl for the Naive Bayes pipeline and lr_model.pkl for the Logistic Regression pipeline, with the benefit of loading pre-trained models instantly without retraining.

### Phase 8: Prediction Interface

The prediction interface implements a function called predict_sentiment that takes text, model, and model_name as parameters. The input is raw review text, the process involves cleaning, vectorizing, predicting, and calculating confidence, and the output is a sentiment label (POSITIVE or NEGATIVE) plus a confidence score.

## 📈 Results & Performance

### Model Comparison

The model comparison shows that Naive Bayes with CountVectorizer achieves 85.0% accuracy with precision of 0.85, recall of 0.85, F1-score of 0.85, and training time of approximately 30 seconds. Logistic Regression with TF-IDF, which is the best model, achieves 88.5% accuracy with precision of 0.88, recall of 0.89, F1-score of 0.88, and training time of approximately 2 minutes.

### Why Logistic Regression Performs Better

Logistic Regression performs better because TF-IDF outperforms Bag-of-Words as word importance weighting helps identify discriminative features. The linear model has flexibility to learn nuanced feature weights rather than just word probabilities. It achieves balanced performance with high precision AND recall so no trade-off is needed. There is no class bias as it performs equally well on positive and negative reviews.

### Confusion Matrix Analysis

The confusion matrix breakdown for Logistic Regression shows that for actual negative reviews, 11,100 were correctly predicted as negative and 1,400 were incorrectly predicted as positive, giving 88.8% recall for negatives. For actual positive reviews, 1,500 were incorrectly predicted as negative and 11,000 were correctly predicted as positive, giving 88.0% recall for positives. Key observations are approximately 1,400 false positives (negative reviews predicted as positive), approximately 1,500 false negatives (positive reviews predicted as negative), and no significant bias toward either class.

### Sample Predictions

Sample predictions demonstrate the model's performance. The review "This movie was absolutely fantastic! Best film ever." with true label Positive is predicted as Positive with 94.3% confidence. The review "Terrible movie. Waste of time and money." with true label Negative is predicted as Negative with 91.2% confidence. The review "It was okay, nothing special but watchable." which is Neutral is predicted as Positive with only 52.8% confidence, showing that low confidence indicates neutral sentiment. The review "I loved every minute! Highly recommend." with true label Positive is predicted as Positive with 96.7% confidence. The review "One of the worst movies I've watched." with true label Negative is predicted as Negative with 89.4% confidence.

## 🚀 Installation & Setup

### Prerequisites

To set up this project, you need Python 3.8 or higher, pip as the Python package manager, 2GB free disk space for the dataset, and Jupyter Notebook.

### Installation Steps

Step 1 is to clone the repository using the command `git clone https://github.com/AbhiRaj067/imdb-sentiment-analysis.git` followed by `cd imdb-sentiment-analysis`. Step 2 is to install dependencies using `pip install -r requirements.txt`, or alternatively install manually using `pip install pandas numpy scikit-learn nltk beautifulsoup4 matplotlib seaborn wordcloud requests joblib`. Step 3 is to download NLTK data by opening a Python terminal and running `import nltk` followed by `nltk.download('stopwords')`, `nltk.download('wordnet')`, and `nltk.download('punkt')`. Step 4 is to launch Jupyter Notebook using the command `jupyter notebook`. Step 5 is to run the notebooks, where for complete analysis you open `imdb_sentiment_analysis_full.ipynb` and for quick start you open `imdb_sentiment_analysis_simple.ipynb`. Note that the dataset will auto-download approximately 80MB on first run and save locally for future use.

## 💻 How to Use

### Full EDA Notebook (Option 1)

The Full EDA Notebook is best for learning, presentations, and comprehensive analysis. The runtime is approximately 10 to 15 minutes for the first run with download. What you get includes step-by-step data loading and preprocessing, visual EDA with class distribution and word clouds, two model implementations (Naive Bayes and Logistic Regression), confusion matrices and detailed metrics, saved model files for reuse, and a custom prediction function. To use it, run all cells to get complete analysis with all visualizations.

### Simplified Notebook (Option 2)

The Simplified Notebook is best for quick results, production pipeline, and using only the best model. The runtime is approximately 5 to 8 minutes. What you get includes the same data loading and preprocessing, single best model (Logistic Regression with TF-IDF), classification report, and ready-to-use prediction function. To use it, run all cells to get an instant sentiment classifier.

### Making Predictions

After running either notebook, you can make predictions on new reviews using the predict_sentiment function. For a positive review example, use `predict_sentiment("Amazing film! Loved the acting and plot twists.", lr_model, "Logistic Regression")` which outputs Prediction: POSITIVE with 92.15% confidence. For a negative review example, use `predict_sentiment("Boring and predictable. Complete waste of money.", lr_model, "Logistic Regression")` which outputs Prediction: NEGATIVE with 87.32% confidence. For a neutral or mixed review example, use `predict_sentiment("Decent movie, not great but not terrible either.", lr_model, "Logistic Regression")` which outputs Prediction: POSITIVE with 54.67% confidence, where the low confidence indicates neutral sentiment.

## 📁 Project Structure

The project structure consists of several key files and directories. At the root level, README.md provides complete project documentation, requirements.txt lists Python dependencies, LICENSE contains the MIT License, and .gitignore contains Git ignore rules using the Python template. The main notebooks are imdb_sentiment_analysis_full.ipynb which is the full version with EDA and 2 models, and imdb_sentiment_analysis_simple.ipynb which is the simplified version with the best model only. The data directory is auto-generated on first run and contains train.csv with 25K parsed training reviews, test.csv with 25K parsed test reviews, the aclImdb directory with raw extracted dataset which is gitignored, class_distribution.png showing class balance visualization, pos_wordcloud.png showing positive sentiment word cloud, and neg_wordcloud.png showing negative sentiment word cloud. The models directory is generated after training in the full version and contains nb_model.pkl with the saved Naive Bayes model and lr_model.pkl with the saved Logistic Regression model. Important notes are that the data folder is created automatically when you run the notebook, large CSV files train.csv and test.csv are in .gitignore and not uploaded to GitHub, you can safely delete the data/aclImdb folder after CSVs are generated to save space, and model .pkl files are also gitignored due to size and can be regenerated by running the full notebook.

## 🔍 Key Features & Highlights

### Full Version Features

The Full Version includes automated data pipeline with one-click download extract and parse from Stanford URL, comprehensive text cleaning with HTML removal using regex stopwords removal and lemmatization, visual EDA with class distribution plots and word clouds for positive versus negative sentiment, dual model comparison between Naive Bayes with CountVectorizer and Logistic Regression with TF-IDF, multiple vectorization approaches showing both Bag-of-Words and TF-IDF representations, rich visualizations using Matplotlib and Seaborn plots with confusion matrix heatmaps, model persistence with save and load functionality using joblib, and detailed evaluation metrics including precision recall F1-score and confusion matrices.

### Simplified Version Features

The Simplified Version includes streamlined pipeline with core functionality only, best model focus on Logistic Regression with TF-IDF as it has the highest accuracy, faster execution at approximately 50% less runtime, production-ready clean minimal code suitable for deployment, same preprocessing with identical text cleaning pipeline as the full version, and interactive prediction with ready-to-use sentiment function.

### Technical Highlights

Both versions implement robust text preprocessing with BeautifulSoup for HTML tag removal, regex for non-alphabetic character filtering, NLTK stopwords removal, and WordNetLemmatizer for word normalization. The scikit-learn pipeline architecture ensures clean reproducible code that chains vectorization and model training. Model comparison capabilities allow empirical testing of different approaches. Confidence scoring provides probabilistic predictions using predict_proba method. The system handles large datasets efficiently by processing 50,000 reviews. It is extensible and modular with clear function definitions and deployment-ready with saved models for production use.

## 📚 What I Learned

### Technical Skills

Through this project I developed several technical skills. In NLP Pipeline Development I learned to build end-to-end text processing workflows from raw data to predictions. For Text Preprocessing I gained experience with HTML cleaning using BeautifulSoup, regex pattern matching for text normalization, NLTK stopword removal, and WordNetLemmatizer for word
