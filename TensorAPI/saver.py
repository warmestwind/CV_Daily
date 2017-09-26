import tensorflow as tf
# Create some variables.

w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
w2 = tf.Variable(tf.truncated_normal(shape=[10]), name='w2')
result=w1+w2
tf.add_to_collection('vars', w1)
tf.add_to_collection('vars', w2)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './my-model',global_step =1000)
# `save` method will call `export_meta_graph` implicitly.
# you will get saved graph files:my-model.meta
