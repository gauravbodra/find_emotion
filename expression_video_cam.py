# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:31:28 2018

@author: admin
"""
import cv2
import numpy as np
from keras.preprocessing import image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']




#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression.json", "r").read())
model.load_weights('facial_expression_weight.h5') 


cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.flip(img,1)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
		 
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (48, 48)) 
        
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 
		
        predictions = model.predict(img_pixels) 
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
		
		
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

	
cap.release()
cv2.destroyAllWindows()
