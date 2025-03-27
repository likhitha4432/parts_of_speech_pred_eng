import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("pos_tag.h5")

# Load vectorizers
with open("tvs.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("tvt.pkl", "rb") as f:
    cv_vectorizer = pickle.load(f)

# Streamlit UI
st.title("Part-of-Speech Prediction")
st.write("Enter a sentence, and the model will predict POS tags.")

# User input
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        # Transform input text
        input_vector = tfidf_vectorizer([user_input])
        
        # Predict
        predicted = model.predict(np.array(input_vector))
        #predicted_indices = np.argmax(prediction, axis=1)  # Assuming categorical output
        
        # Convert indices to words
        out=[]
        for y in np.argmax(predicted[0],axis=1):
            out.append(cv_vectorizer.get_vocabulary()[y])
        
        print(user_input," ".join(out))
        
        st.success(f"Predicted POS Tags: {' '.join(out)}")
    else:
        st.warning("Please enter some text.")
