import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset =20
imgSize = 300

labels = ["6", "7", "8"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            resized_img = cv2.resize(imgCrop, (wCal, imgSize))

            w_offset = math.ceil((imgSize - wCal) / 2)
            img_white[:, w_offset:wCal + w_offset] = resized_img
            prediction, index = classifier.getPrediction(img_white)
            print(prediction)
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            resized_img = cv2.resize(imgCrop, (imgSize, hCal))
            h_offset = math.ceil((imgSize - hCal) / 2)
            img_white[h_offset:hCal+h_offset, :] = resized_img
            prediction, index = classifier.getPrediction(img_white)
            print(prediction)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x + w + offset, y - offset), (255, 0, 255), thickness=-1)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

        # cv2.imshow("img_white", img_white)

    cv2.imshow("imgOutput", imgOutput)
    cv2.waitKey(1)


