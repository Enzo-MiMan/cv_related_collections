import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

folder = "./images/eight"

offset =20
imgSize = 300
counter = 0

while True:
    success, img = cap.read()
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
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            resized_img = cv2.resize(imgCrop, (imgSize, hCal))
            h_offset = math.ceil((imgSize - hCal) / 2)
            img_white[h_offset:hCal+h_offset, :] = resized_img

        cv2.imshow("img_white", img_white)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(os.path.join(folder, f'{counter}.jpg'), img_white)
        print(counter)

