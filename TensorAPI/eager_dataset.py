import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

'''
tfe.enable_eager_execution()

# tfe iter
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.shuffle(5).repeat(2)
for one_element in tfe.Iterator(dataset):
    print(one_element)
iter = tfe.Iterator(dataset) 
print(iter.next())

'''
'''
g = tf.Graph()
with g.as_default():
    a=tf.constant(3)

sess=tf.Session(graph=g)

print(sess.run(a))

'''
# sess iter : out of range
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.shuffle(5).repeat(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
#'''
