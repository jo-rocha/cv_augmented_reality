import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
targetImage = cv.imread('images/target.jpeg')
kiliImage = cv.imread('images/kili.jpg')
#resizing the image that is going to show up on top of the target image to be the same size as the target image("qrcode")
hT, wT, cT = targetImage.shape
kiliImage = cv.resize(kiliImage, (wT, hT))

#detector
orb = cv.ORB_create(nfeatures = 1000)
kp1, des1 = orb.detectAndCompute(targetImage, None)
targetImage = cv.drawKeypoints(targetImage, kp1, None)



#webcam loop
while cap.isOpened() and cv.waitKey(1) == -1:
    success, frame = cap.read()
    kp2, des2 = orb.detectAndCompute(frame, None)
    # frame = cv.drawKeypoints(frame, kp2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    cv.imshow("altered", kiliImage)
    cv.imshow("target", targetImage)
    cv.imshow('webcam', frame)
