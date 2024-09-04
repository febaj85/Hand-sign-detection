import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time
import sys
import os

cap = cv2.VideoCapture(0)  # 0 is the ID number of your webcam
detector = HandDetector(maxHands=2)  # Max hands to detect
offset = 20
imgsize = 300
counter = 0

# Specify the folder where images should be saved
folder = r'C:\Users\febaj\Downloads\_________________LEARNINGSS____________\3D_handsign\cv\data\C'
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
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
        else:
            k = imgsize / width
            hcal = int(np.ceil(k * height))
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hGap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hGap:hcal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # 1 millisecond delay
    if key == ord('s'):
       
        filepath = os.path.join(folder, f'Image_{int(time.time()*1000)}.jpg')
        cv2.imwrite(filepath, imgwhite)
        counter += 1
        print(f"Saved {filepath}")
