#!/usr/bin/env python
# coding: utf-8

# In[11]:


import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd 
import numpy as no 
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# In[ ]:


st.title("Text Message Classifier")
message_text = st.text_input("Enter a message for spam evaluation")


model = joblib.load('spam_classifier.joblib')


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    text = text.lower()
    
        
    #Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)

def classify_message(model, message):
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])
    return {"Type of Message": label, "Probability of Spam": spam_prob[0][1]}

#output the models predictions as a dictionary 

if message_text != "":
    result = classify_message(model, message_text)
    st.write(result)
    
    explain_pred = st.button("Explain Predictions")
     
    if explain_pred: 
        with st.spinner("Generating explanations"):
            class_names = ['ham', 'spam']
            explainer = LimeTextExplainer(class_names = class_names)
            exp= explainer.explain_instance(message_text,
                                        model.predict_proba, num_features =10)
            components.html(exp.as_html(), height = 800)
                