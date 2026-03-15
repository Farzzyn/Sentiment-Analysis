# 🗣️ Customer Review Sentiment Analyzer

A lightweight AI-powered customer review analyzer that classifies text reviews into **Positive**, **Neutral**, or **Negative** sentiment. Users can paste any review into a simple web interface and instantly receive a sentiment prediction powered by an NLP model trained on real customer feedback.

**Mission:** Turn unstructured reviews into clear, actionable sentiment insights.

## Features

- **Automatic Sentiment Classification**: Uses Machine Learning (SVM/Logistic Regression) to categorize feedback.
- **Text Preprocessing Pipeline**: Automatically cleans reviews (lowercase, removes punctuation and stopwords, tokenizes, and lemmatizes).
- **TF-IDF Vectorization**: Converts text into machine learning features for fast and effective NLP.
- **Simple Web Interface**: Powered by Streamlit, users can enter a review and instantly view the predicted sentiment.

## Tech Stack

- **Programming Language:** Python
- **Data Processing:** pandas, numpy
- **NLP Processing:** nltk
- **Machine Learning:** scikit-learn
- **Deployment:** Streamlit

## Getting Started

### Prerequisites

Ensure you have Python installed. The required packages are listed in `requirements.txt`.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Farzzyn/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

You can run the web application using the provided batch script or directly via Streamlit:

**Using the batch script (Windows):**
```bash
run.bat
```

**Using Streamlit directly:**
```bash
streamlit run app.py
```

The app will start on an accessible local URL, typically `http://localhost:8501`.

## Project Structure

- `app.py`: The Streamlit web application.
- `sentiment_pipeline.py`: Contains data training, text preprocessing, and saved ML models generation.
- `masterplan.md`: Details the project mission, roadmap, and design principles.
- `dataset.csv` / `Mall_Customers.csv`: Datasets used for demonstration and model training.

## Roadmap

- **Phase 1**: MVP with EDA, basic text preprocessing, TF-IDF features, and ML model evaluation.
- **Phase 2**: Model Optimization using hyperparameter tuning and cross-validation.
- **Phase 3**: Deployment of the Streamlit app.
- **Phase 4**: Future enhancements like real-time review analytics, API integration, and batch review tools.
