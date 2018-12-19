import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

g = tf.Graph()
with g.as_default():
    a = tf.get_variable("v_a", shape=(10,10,3), initializer = tf.initializers.constant(np.ones((10,10,3))))

    size = get_num_params()
    print(size)
