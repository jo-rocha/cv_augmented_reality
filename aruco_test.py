import numpy as np
import argparse
import cv2
import sys

video = ('images/video.mp4')
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
arucoParams = cv2.aruco.DetectorParameters_create()
#
captureVideo = cv2.VideoCapture(video)
success, image = captureVideo.read()
cv2.imshow('first frame', image)
cv2.waitKey(0)
#
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)