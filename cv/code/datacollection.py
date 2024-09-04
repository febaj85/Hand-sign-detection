import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
from cvzone.ClassificationModule import Classifier
import time
import tensorflow.keras
import sys
import os
import tensorflow as tf
print(tf.__version__)

cap = cv2.VideoCapture(0)  # 0 is the ID number of your webcam
detector = HandDetector(maxHands=2)  # Max hands to detect
classifier= Classifier(MODEL_PATH,LABEL_PATH)
offset = 20
imgsize = 300
counter = 0
labels=['A','B','C']
# Specify the folder where images should be saved
folder = <IMAGE_SAVING_FOLDER_PATH>
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    imgoutput=img.copy()
    hands, img = detector.findHands(img)#for not drawing the image draw=False
    
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']  # Bounding box

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + height + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + width + offset)
        imgCrop = img[y1:y2, x1:x2]
        cv2.imshow("Imagecrop", imgCrop)

        aspectratio = height / width
        if aspectratio > 1:
            k = imgsize / height
            wcal = int(np.ceil(k * width))
            imgResize = cv2.resize(imgCrop, (wcal, imgsize))
            wGap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wGap:wcal + wGap] = imgResize
            prediction,index=classifier.getPrediction(imgwhite,draw=False)
            print(prediction,index)
        else:
            k = imgsize / width
            hcal = int(np.ceil(k * height))
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hGap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hGap:hcal + hGap, :] = imgResize
            prediction,index=classifier.getPrediction(imgwhite,draw=False)

        cv2.putText(imgoutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgoutput,(x-offset,y-offset),(x+width+offset,y+height+offset),(255,0,255),4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)

    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)  # 1 millisecond delay
