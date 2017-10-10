import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

x = tf.placeholder(tf.float32, shape=[None, 9,2])
shape = x.get_shape().as_list()        # a list: [None, 9, 2]
dim = numpy.prod(shape[1:])            # dim = prod(9,2) = 18
x2 = tf.reshape(x, [-1, dim])           # -1 means "all"

#or tensorflow higher api
input_image=tf.contrib.layers.flatten(x)