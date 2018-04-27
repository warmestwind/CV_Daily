#https://tensorflow.google.cn/versions/r1.5/api_docs/python/tf/layers/Dense#trainable_weights
import tensorflow as tf

he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

data = tf.constant([[1,2.0,3,4,5]])
num_hidden = 10
hidden = tf.layers.dense(data, num_hidden, activation=tf.nn.elu,
        kernel_initializer=he_init,
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(),
        name= "model" + "hidden1")

hidden_class = tf.layers.Dense(num_hidden, activation=tf.nn.elu,
        kernel_initializer=he_init,
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(),
        name= "Dense")

w = hidden_class.trainable_weights
name = hidden_class.name

out = hidden_class.apply(data)

sess =tf.Session()
init = [tf.local_variables_initializer() ,tf.global_variables_initializer()]

#input_data = sess.run(data)
sess.run(init)
output_data =  sess.run(hidden)

print(sess.run(out))
print(w)
print(sess.run(w))
print(output_data)

