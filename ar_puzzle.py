import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
targetImage = cv.imread('images/export-qr-codes/target6.png')
kiliImage = cv.imread('images/kili.jpg')
#resizing the image that is going to show up on top of the target image to be the same size as the target image("qrcode")
hT, wT, cT = targetImage.shape
kiliImage = cv.resize(kiliImage, (wT, hT))

#detector
orb = cv.ORB_create(nfeatures = 1000)
kp1, des1 = orb.detectAndCompute(targetImage, None)#detecta pontos chave na imagem alvo
targetImage = cv.drawKeypoints(targetImage, kp1, None)#para printar os pontos destacados na imagem


#webcam loop
while cap.isOpened() and cv.waitKey(1) == -1:
    success, frame = cap.read()
    kp2, des2 = orb.detectAndCompute(frame, None)#detecta pontos chave na imagem da webcam
    frame = cv.drawKeypoints(frame, kp2, None)#destaca os pontos detectados na imagem da webcam

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)#encontra o número de matches entre os pontos da imagem alvo e a da webcam
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))#quantas matches que eu tenho que posso utilizar
    imgFeatures = cv.drawMatches(targetImage, kp1, frame, kp2, good, None, flags=2)#desenha os pontos semelhantes entre o alvo e a imagem da webcam

    # q

    # cv.imshow('squareFrame', img2)
    cv.imshow('matches', imgFeatures)
    cv.imshow("altered", kiliImage)
    cv.imshow("target", targetImage)
    cv.imshow('webcam', frame)
