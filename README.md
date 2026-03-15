# 🗣️ Customer Review Sentiment Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A lightweight AI-powered customer review analyzer that classifies text reviews into **Positive**, **Neutral**, or **Negative** sentiment. Turn unstructured reviews into clear, actionable sentiment insights instantly!

</div>

---

## 📑 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 About the Project

Businesses receive thousands of customer reviews across platforms. Manually reading and understanding sentiment is slow and inconsistent. This NLP system solves that by automatically analyzing review sentiment and classifying feedback into Positive, Neutral, or Negative. 

Targeted for data scientists, product managers, and small businesses looking to quickly understand customer perception.

## ✨ Key Features

- **Automated Sentiment Classification**: Categorizes user feedback instantly.
- **Robust Text Preprocessing**: Automatically cleans text (lowercasing, punctuation/stopword removal, tokenization, lemmatization).
- **TF-IDF Vectorization**: Transforms textual data into rich machine learning features.
- **Multiple ML Models**: Support for evaluating Logistic Regression, Naive Bayes, SVM, and Random Forest.
- **Interactive UI**: A simple and responsive web interface powered by Streamlit.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **NLP & Processing:** `nltk`
- **Machine Learning:** `scikit-learn`
- **Web Framework:** `Streamlit`

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have Python installed on your system.
* Python 3.8 or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Farzzyn/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

You can run the application directly using Streamlit or by using the provided batch file (on Windows).

**Option 1: Using the batch script (Windows only)**
```bash
run.bat
```

**Option 2: Using Streamlit**
```bash
streamlit run app.py
```

Once running, open your browser and navigate to `http://localhost:8501`. Simply paste a customer review into the text area and click **"Predict Sentiment"** to see the results!

## 📂 Project Structure

```text
Sentiment-Analysis/
├── app.py                   # Streamlit web application
├── sentiment_pipeline.py    # Data loading, preprocessing, and model training
├── masterplan.md            # Detailed project mission, roadmap, and design principles
├── requirements.txt         # Project dependencies
├── run.bat                  # Batch script to easily run the app on Windows
├── dataset.csv              # Sample dataset for training/testing
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## 🗺️ Roadmap

- [x] **Phase 1 (MVP):** Exploratory Data Analysis (EDA), basic text preprocessing, TF-IDF features, and ML model evaluation.
- [ ] **Phase 2 (Optimization):** Improve accuracy via hyperparameter tuning and cross-validation.
- [x] **Phase 3 (Deployment):** Deploy the model using a Streamlit web app.
- [ ] **Phase 4 (Enhancements):** Real-time review analytics, API integration, and batch review tools.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/Farzzyn">Farzzyn</a></p>
</div>
