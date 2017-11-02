import tensorflow as tf
sess=tf.Session()
#共享变量通过 variable_scope +get_variable 来实现
# 注意， bias1 的定义方式
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3],initializer=tf.constant_initializer())
    bias1 = tf.Variable([0.52], name='bias') #variable 不可以共享

print (Weights1.name)
print (bias1.name)
print (bias1.get_shape())
# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')

  #  bias2 = tf.get_variable('bias')  # ‘bias
weight3=tf.Variable(0.0,name='w')
weight3+=Weights2
sess.run(tf.global_variables_initializer())
print (Weights2.name)
print (sess.run(weight3))
#print (bias2.name)
