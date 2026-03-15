import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

class SentimentPipeline:
    def __init__(self, csv_path):
        """
        Initialize the pipeline with the path to the dataset.
        Expected CSV columns: 'review_text', 'sentiment_label' (or 'rating' to infer sentiment)
        """
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Support Vector Machine': SVC(kernel='linear', random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""

    def load_data(self):
        """Load the dataset from the specified CSV file."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        # Check if expected columns are present
        if 'review_text' not in self.df.columns or 'sentiment_label' not in self.df.columns:
            print("Warning: Ensure your CSV has 'review_text' and 'sentiment_label' columns.")
            print(f"Current columns found: {self.df.columns.tolist()}")
            
        print(f"Dataset Shape: {self.df.shape}")
        
    def perform_eda(self):
        """Perform Exploratory Data Analysis (EDA)"""
        print("\n--- Performing EDA ---")
        
        if 'sentiment_label' in self.df.columns:
            # 1. Sentiment Distribution
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.df, x='sentiment_label')
            plt.title("Sentiment Distribution")
            plt.show()

        if 'review_text' in self.df.columns:
            # 2. Review Length Analysis
            self.df['review_length'] = self.df['review_text'].astype(str).apply(len)
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df['review_length'], bins=50)
            plt.title("Review Length Distribution")
            plt.show()

            # 3. Word Cloud
            text = " ".join(self.df['review_text'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Word Cloud of Reviews")
            plt.show()

    def clean_text(self, text):
        """
        Text preprocessing steps:
        - lowercase conversion
        - punctuation removal
        - stopword removal
        - tokenization
        - lemmatization
        """
        text = str(text).lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stopword removal & Lemmatization
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return " ".join(clean_tokens)

    def preprocess_data(self):
        print("\n--- Preprocessing Text ---")
        self.df['clean_text'] = self.df['review_text'].apply(self.clean_text)
        print("Sample of cleaned text:")
        print(self.df[['review_text', 'clean_text']].head())

    def feature_engineering(self):
        """Convert text into machine learning features using TF-IDF."""
        print("\n--- Feature Engineering (TF-IDF) ---")
        X = self.vectorizer.fit_transform(self.df['clean_text'])
        y = self.df['sentiment_label']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train samples: {self.X_train.shape[0]}, Test samples: {self.X_test.shape[0]}")

    def train_and_evaluate_models(self):
        print("\n--- Training & Evaluating Models ---")
        best_score = 0
        
        for name, model in self.models.items():
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            print(f"\nModel: {name}")
            print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
            print("-" * 30)
            
            if acc > best_score:
                best_score = acc
                self.best_model = model
                self.best_model_name = name

        print(f"\n🏆 Best Model: {self.best_model_name} with Accuracy ~ {best_score:.4f}")

    def run_pipeline(self):
        """Run the full end-to-end Phase 1 pipeline"""
        self.load_data()
        # Ensure we have the necessary columns before proceeding
        if 'review_text' in self.df.columns and 'sentiment_label' in self.df.columns:
            # Uncomment the next line to show EDA plots during run
            # self.perform_eda() 
            self.preprocess_data()
            self.feature_engineering()
            self.train_and_evaluate_models()
        else:
            print("Cannot run full pipeline until 'review_text' and 'sentiment_label' are provided in the dataset.")

if __name__ == "__main__":
    # Example Usage: Replace with the actual CSV path
    # Setting up the path logic for the given CSV
    csv_file = "dataset.csv"  # The dataset
    
    pipeline = SentimentPipeline(csv_file)
    pipeline.run_pipeline()
