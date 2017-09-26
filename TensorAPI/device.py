#coding=utf-8
import tensorflow as tf

# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

'''
config = tf.ConfigProto()
config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)
'''

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))
