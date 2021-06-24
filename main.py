import tensorflow as tf
import numpy as np
import pickle5 as pickle
import streamlit as st
import re
from PIL import Image
import random

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(string_review):
    # Remove all the special characters
    string_process = re.sub(r'\W', ' ', str(string_review))
    # remove all single characters
    string_process = re.sub(r'\s+[a-zA-Z]\s+', ' ', string_process)
    # Remove single characters from the start
    string_process = re.sub(r'\^[a-zA-Z]\s+', ' ', string_process) 
    # Substituting multiple spaces with single space
    string_process = re.sub(r'\s+', ' ', string_process, flags=re.I)
    # Removing prefixed 'b'
    string_process = re.sub(r'^b\s+', '', string_process)
    # Converting to Lowercase
    string_process = string_process.lower()
    return string_process

def import_and_predict(string_review, model, tokenizer):
    string_cleaned = clean_text(string_review)
    text_token = tokenizer.texts_to_matrix([string_cleaned], mode="tfidf")
    prediction = nn_model.predict(text_token)
    prediction = prediction.argmax(axis=-1)
    return prediction

# Load the NN Model
nn_model = tf.keras.models.load_model("sentiment.h5")

# Load the Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    # text_tokenizer = Tokenizer(num_words=512)
    text_tokenizer = pickle.load(handle)

st.title("Sentiment Analysis App")

st.header("This is a simple sentiment analysis application based on Amazon Fine Food Review Dataset")

st.write("You can try to write a review text and get the sentiment prediction of your review text!")

image_path = "image/" + str(random.randint(1, 3)) +".jpg"
image = Image.open(image_path)
st.image(image, width = 500, caption='Product Image Example')

string_review = st.text_area("Write a review text, CTRL+Enter to Predict", help="You can write some review text and get the sentiment of your review")

if string_review:
    st.subheader("Your Review Text")
    st.write(string_review)
    
    prediction = import_and_predict(string_review, nn_model, text_tokenizer)
    # print(prediction.shape)
    
    st.subheader("Sentiment Prediction")

    if prediction == 0:
        st.markdown('Sentiment : **Negative** :disappointed:')
    elif prediction == 1:
        st.markdown('Sentiment : **Neutral** :neutral_face:')
    else:
        st.markdown('Sentiment : **Positive** :smile:')

# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)

#     st.image(image, use_column_width=True)
    
#     prediction = import_and_predict(image, model)
    
#     if np.argmax(prediction) == 0:
#         st.write("Result : Covid-19")
#     elif np.argmax(prediction) == 1:
#         st.write("Result : Normal")
#     else:
#         st.write("Result : Viral Pneumonia")

#     st.text("Probability (0: Covid-19, 1: Normal, 2: Viral Pneumonia)")
#     st.write(prediction)