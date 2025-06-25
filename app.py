import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Cervical Cancer Risk Predictor", layout="centered")
st.title("🧬 Cervical Cancer Risk Predictor")
st.markdown("Upload your dataset or enter details to estimate risk level.")

uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    if st.button("Train Model and Predict"):
        if 'Biopsy' in df.columns:
            X = df.drop('Biopsy', axis=1)
            y = df['Biopsy']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            st.success(f"Model trained with accuracy: {score:.2f}")
        else:
            st.error("Dataset must include a 'Biopsy' column for prediction.")

st.caption("This is a demo app. For real diagnosis, consult a medical professional.")