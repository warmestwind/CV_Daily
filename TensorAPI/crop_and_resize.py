import tensorflow as tf
import numpy as np

image = np.array([[1,2,3,4,5,6],
                  [7,8,9,10,11,12],
                  [1,2,3,4,5,6], 
                  [7,8,9,10,11,12]])
image2 = image[np.newaxis,:,:,np.newaxis]
print(image2)
print("source is ",image2.shape)

bbox = np.array([[0, 0, 1, 1 ]])
bbox = tf.convert_to_tensor(bbox, tf.float32)
print(bbox)
crop = tf.image.crop_and_resize(image2, bbox, [0], [4,6])
print(crop)
sess= tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(crop))
