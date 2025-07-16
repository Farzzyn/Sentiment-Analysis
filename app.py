
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


st.title(" 🌀 Customer Review Sentiment Analyzer")
user_input = st.text_area("Enter a customer review:")


if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        elif prediction == 0:
            st.info("😐 Neutral Review")
        else:
            st.error("❌ Negative Review")
    else:
        st.warning("Please enter some text.")
