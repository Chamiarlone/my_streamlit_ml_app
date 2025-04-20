#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Apr 20 13:52:59 2025

@author: Chalermchai
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

#load model
with open ('model.pkl','rb') as f:
    model = pickle.load(f)
    
#set title application
st.title("Image Classification with MobileNetV2 by Chalermchai Nichee 6531501015")

#flie upload
upload_file = st.file_uploader("Upload image:", type=["jpg","jpeg","png"])

if upload_file is not None:
    #display image on screen
    img = Image.open(upload_file)
    st.image(img, caption="Upload Image")
    
    #preproccessing
    img = img.resize ((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    
    #prediction
    preds = model.predict(x)
    top_pred = decode_predictions(preds, top=3)[0]
    
    #display priediction
    st.subheader("Prediction:")
    for i,pred in enumerate(top_pred):
        st.write(f"{i+1}. **{pred[1]}** - {round(pred[2]*100,2)}%")