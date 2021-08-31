# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 20:32:51 2021

@author: gta4s
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32)
    
    return faces,gray_img

def trainingDataLabels(directory):
    faces = []
    faceID = []
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            idx = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print(idx,img_path)
            test_img = cv2.imread(img_path)
            if test_img is None:
                
                continue
            faces_rect,gray_img = faceDetection(test_img)
            [x,y,w,h] = faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(idx))
    
    return faces,faceID


def trainModel(faces,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0))
    

def put_text(test_img,text,x,y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX,1, (255,0,0),1)


    
test_image = cv2.imread('z.jpg')
faces_detected, gray_img = faceDetection(test_image)

faces,faceID = trainingDataLabels('C:/Users/gta4s/OneDrive/Desktop/TASK2_HBPF/Training/')
face_recognizer = trainModel(faces, faceID)

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+w]
    label,confidence = face_recognizer.predict(roi_gray)
    draw_rect(test_image, face)
    ans = 'Person '+str(label)
    if confidence > 65:
        ans = 'Unknown'
        
    #ns = str(confidence)
    #ans = 'P:'+str(label) +' C:'+str(confidence)
    put_text(test_image, ans, x, y)
    
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
subplot.imshow(test_image)
    

"""
test_image = cv2.imread('9336923.1.jpg')
faces_detected, gray_img = faceDetection(test_image)
print("faces detected ", faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_image, (x,y), (x+w,y+h), (255,0,0))

fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
subplot.imshow(test_image)
"""

