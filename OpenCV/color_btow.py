import cv2
import numpy as np

img = cv2.imread("n.jpg")
print(img.shape)
img2 = img.copy()
for i in range(413): #450
    for j in range(295):# 350
        pix = img[i,j,:]
        #if (pix[0]<200 and pix[0]>100 and pix[1]>60 and pix[1]<120 and pix[2]< 40 ):
        if (pix[0] < 256 and pix[0] > 100 and pix[1] > 60 and pix[1] < 255 and pix[2] < 60):
            #print("find")
            img2[i,j,:] = [255,255,255]
img2 = cv2.GaussianBlur(img2, (3,3), 0.1)
#img2 = cv2.medianBlur(img2, 3)
# for i in range(413): #450
#     for j in range(295):# 350
#         pix2 = img2[i,j,:]
#         if  np.array_equal(pix2, np.array([255,255,255])):
#                 #print("find")
#                 img[i,j,:] = np.array([255,255,255])
cv2.imshow("src",img)
cv2.imshow("dst",img2)
cv2.imwrite("dst.jpg",img2)
cv2.imwrite("src.jpg",img)
cv2.waitKey()

# img = cv2.imread("src.jpg")
# img2 = cv2.GaussianBlur(img, (5,5), 0.05)

# cv2.imwrite("final.jpg",img2)
# cv2.waitKey()
