import cv2 as cv
import sys

img = cv.imread('images/kili.jpg')
cv.imshow('show', img)
cv.waitKey()
