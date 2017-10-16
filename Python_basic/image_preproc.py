import cv2

img=cv2.imread('red10.jpg',1)
img[0,0]=[255,0,0]
img[1,1]=[0,0,0]
#cv2.imshow('image',img)
cv2.imwrite('red11.jpg',img)
print(img[0,0,2])
#[x,y,z]  x: row  y: column z:index of color component
img_rgb = img[:, :, ::-1]
# ::-1 ： reverse the component
print(img_rgb[:,:])
#cv2.waitKey(0)
