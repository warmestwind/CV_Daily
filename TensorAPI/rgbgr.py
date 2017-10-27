import tensorflow as tf
import os

with tf.Session() as sess:
    img = tf.zeros([3,2],tf.int32)
    print(sess.run(tf.shape(img)))
    red, green, blue = tf.split(img, 3, 0) #split s along 3rd axis
    assert red.get_shape().as_list() == [1, 2] ,'尺寸不对' #tensorflow api
    print(red.get_shape()[0:]) #return tuple(1,2) , as_list() ->[1, 2]  [0:] start from 0rd element 
    print(red.get_shape()[0].value) # -> 1
    tuple1=(1,2)
    list1=list(tuple1) #python api
    print("list1",list1)

    bgr = tf.concat(axis=0, values=[
            blue ,
            green,
            red])
    print(bgr.get_shape()[0:]) # ->（3，2）

    logs_dir='F:\\SourceCode\\Python\\test\\log'
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
