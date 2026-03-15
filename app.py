import streamlit as st
from sentiment_pipeline import SentimentPipeline

# Set up page configurations
st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_and_train_model():
    """Load data and train the best model. Uses caching to only do this once."""
    pipeline = SentimentPipeline("dataset.csv")
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.feature_engineering()
    pipeline.train_and_evaluate_models()
    return pipeline

def main():
    st.title("🗣️ Customer Review Sentiment Analyzer")
    st.markdown("""
    This tool categorizes feedback into **Positive**, **Neutral**, or **Negative** sentiments.
    """)

    with st.spinner("Initializing models..."):
        pipeline = load_and_train_model()

    st.success(f"Models loaded successfully! Best model logic: **{pipeline.best_model_name}**")

    st.divider()
    
    st.subheader("Predict Sentiment")
    user_input = st.text_area("Enter a customer review below:", placeholder="E.g., The product was amazing and delivery was fast!")
    
    if st.button("Predict Feeling", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Clean text
                cleaned = pipeline.clean_text(user_input)
                # Feature engineering
                features = pipeline.vectorizer.transform([cleaned])
                # Predict
                prediction = pipeline.best_model.predict(features)[0]
                
                # Output result mapping
                if prediction == 'Positive':
                    st.success("### Sentiment: Positive 😊")
                elif prediction == 'Negative':
                    st.error("### Sentiment: Negative 😠")
                else:
                    st.info("### Sentiment: Neutral 😐")
                
                import pandas as pd
                st.write("**Cleaned Input Features Used For Prediction:**")
                st.code(cleaned)

if __name__ == "__main__":
    main()
