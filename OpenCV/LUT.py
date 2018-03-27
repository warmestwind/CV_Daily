import cv2
import numpy as np

img = np.zeros((512,512,1), np.uint8)

for i in range(0,512):
    for j in range(0,512):
        img[i][j]= j/2

lut = np.zeros((1,256,3), np.uint8)
lut[0][25] = ((0,255,255))

img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_o = cv2.LUT(img_c, lut)


cv2.imshow('i',img)
cv2.imshow('o',img_o)

cv2.waitKey(0)
