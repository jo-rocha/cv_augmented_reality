import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os
import sys

def loadAugmentedImages(path):
    myList = os.listdir(path)
    augDict = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv.imread(f'{path}/{imgPath}')
        augDict[key] = imgAug
    return augDict

def findArucoMarkers(img, markerSize = 7, totalMarkers = 50, draw = True):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cria os atributos para a funcao arucoDict
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    #retorna as bounding boxes, os ids dos identifiados, e se houver algum detectado e não identificado
    bboxs, ids, rejectedMarkers = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAugment, drawId = False):
    #define as corners do marcador encontrado
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAugment.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])

    matrix, _ = cv.findHomography(pts2, pts1)
    imgOut = cv.warpPerspective(imgAugment, matrix, (img.shape[1], img.shape[0]))
    #pegar na imagem original apenas a parte que tem o marcador e "pinta de preto"
    # para fazer o overlay apenas em cima do marcador na imagem original(se não fica tudo preto)
    cv.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img + imgOut
    return imgOut


def main():
    video = cv.VideoCapture('video/video.mp4')
    imgAug = cv.imread('images/sheep.jpg')
    
    #captura os frames do video
    fps = int(1000 / video.get(cv.CAP_PROP_FPS))

    cv.namedWindow('open video', cv.WINDOW_GUI_EXPANDED )

    while video.isOpened():
        sucess, frame = video.read()  
        arucoFound = findArucoMarkers(frame)
        #loop entre os markers e augmenta cada um
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                frame = augmentAruco(bbox, id, frame, imgAug)

        # if video finished or no Video Input
        if not sucess:
            break  
        cv.imshow('open video', frame)

        # press 'Q' if you want to exit
        if cv.waitKey(fps) == ord('q'):
            break

if __name__ == '__main__':
    main()