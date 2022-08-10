#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load json and create model
import os
import numpy as np
import cv2
import pickle
import pandas as pd
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from flask import Flask, request, jsonify, render_template


# In[2]:


app=Flask(__name__)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


# In[ ]:


#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# In[ ]:


@app.route('/')
def home():
    return render_template('template.html')


# In[ ]:


@app.route('/detect',methods=["POST"])
def detect():
    """
    For rendering results on HTML GUI
    """
    
    x  = file.filename
    
    full_size_image = cv2.imread(x)
    gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3  , 10)
    
    #detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        emotion= loaded_model.predict(cropped_img)
        emotion_detected = labels[int(np.argmax(emotion))]
    
    return render_template('template.html', detected_expression = 'Detected Emotion is   {}'.format(float(emotion_detected)))  

