import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

# Load the model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Function to clean the input text
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Function to predict spam or ham
def predict_spam(message):
    cleaned_message = clean_text(message)
    transformed_message = tfidf.transform([cleaned_message])
    prediction = model.predict(transformed_message)
    return prediction[0]

# Title of the app
st.title('Spam Message Classifier')

# Add a description to help users understand the app
st.markdown("""
    **Welcome to the Spam Message Classifier App!**
    
    This app uses machine learning to predict whether a message is **spam** or **ham** (non-spam).
    Enter a message in the box below and click 'Enter' to check if it's spam or not.
""")

# Input field for the message
user_input = st.text_input("Enter the message")

# If there's user input, classify the message and display the result
if user_input:
    prediction = predict_spam(user_input)
    if prediction == 'spam':
        st.markdown("🚫 **The message is spam!**")
    else:
        st.markdown("✅ **The message is ham.**")

# Optional: Add a footer or extra information
st.markdown("""
    ---
    Made By Yashwant K.
""")
