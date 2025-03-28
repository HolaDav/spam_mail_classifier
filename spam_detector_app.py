import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

# Set page config
st.set_page_config(page_title="Spam/Ham Classifier", layout="wide")

# Title and description
st.title("📧 Spam/Ham Email Classifier")
st.write("""
This app uses a Logistic Regression model to classify emails as Spam or Ham.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This model was trained on 5572 emails with 97.5% accuracy.
""")

# Load model and vectorizer (we'll create these files next)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'spam_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_artifacts()

# Input text area
user_input = st.text_area("Enter the email text you want to classify:", 
                         "Want 2 get laid tonight? Want real Dogging locations sent direct 2 ur mob?...")

# Prediction button
if st.button('Classify Email'):
    if model and vectorizer:
        # Transform user input
        input_transformed = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)
        
        # Display results
        st.subheader("Result")
        
        if prediction[0] == 'ham':
            st.success(f"✅ This email is **HAM** (not spam)")
            st.metric("Confidence", f"{prediction_proba[0][0]*100:.2f}%")
        else:
            st.error(f"❌ This email is **SPAM**")
            st.metric("Confidence", f"{prediction_proba[0][1]*100:.2f}%")
        
        # Show probability distribution
        st.write("Probability distribution:")
        proba_df = pd.DataFrame({
            'Class': ['Ham', 'Spam'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(proba_df.set_index('Class'))
    else:
        st.error("Model not loaded. Cannot make predictions.")

# Add sample emails for quick testing
st.subheader("Try these examples:")
col1, col2 = st.columns(2)

with col1:
    st.write("**Spam Example:**")
    st.code("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.")

with col2:
    st.write("**Ham Example:**")
    st.code("Hey, are we still meeting for lunch tomorrow at 1pm?")