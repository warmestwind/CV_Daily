import cv2
import numpy as np
img=cv2.imread('img22.jpg',1)
img2=img.copy()
img[0,0]=[255,0,0]
img[1,1]=[0,0,0]
img2[0,0]=[0,0,0]
img2[0,1]=[0,0,0]
img2[1,0]=[0,0,0]
img2[1,1]=[0,0,0]
#cv2.imshow('image',img)
#cv2.imwrite('red11.jpg',img)
#print(img[0,0,2])
img_rgb = img[:, :, ::-1]
print(img_rgb.shape) # (2,2,3）
print(img_rgb[0,0]) # (2,2,3）
#print(np.mean(img[:,:,0]))
#print((np.mean(img2[:,:,0])+np.mean(img[:,:,0]))/2)


#cv2.waitKey(0)
