import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

# at pro begin
tf.enable_eager_execution()

w = tfe.Variable([[2.0]])
w2 = tfe.Variable([[0.0]])

# must define a func
def sin(w2):
   return tf.sin(w2)

with tf.GradientTape() as tape:
  #tape.watch(w) # trace w
  loss = w * w + sin(w2) 
  
grad = tape.gradient(loss, [w,w2])  

print(grad)

# [<tf.Tensor: id=30, shape=(1, 1), dtype=float32, numpy=array([[4.]], dtype=float32)>,
# <tf.Tensor: id=27, shape=(1, 1), dtype=float32, numpy=array([[1.]], dtype=float32)>]
