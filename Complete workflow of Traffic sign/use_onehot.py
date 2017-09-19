import tensorflow as tf
import numpy as np

#x = tf.placeholder(tf.float32, [None, 5,5])
with tf.Session() as sess:
    index=[0,1,2,3,4]
    index_onehot=tf.one_hot(index,5)
    print("one hot label-tensor:",index_onehot.eval())
    index_onehot2=np.asarray(index_onehot)
    #sess.run(index_onehot2)
    print('one hot label-array:', index_onehot2)
    print(type(index_onehot))
    print(type(index_onehot2))

    #print(x.eval(feed_dict={x:index_onehot}))
    #CANT DEED ONEHOT BECAUSE IT IS A TENSOR

   
