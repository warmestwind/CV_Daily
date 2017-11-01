import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
learning_rate = 0.1 
training_epochs = 5 
x_train = np.linspace(-1, 1, 101) 
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33 

file_object=open('x.txt','r')
'''
for num in y_train:
    file_object.write(str(num))
    file_object.write('\n')
file_object.close()
'''
y_list=[]
for line in file_object:
    #print(line)
    y_list.append(line)
#print(len(y_list))
print("shape=",np.shape(y_list))


X = tf.placeholder("float") 
Y = tf.placeholder("float") 
def model(X, w): 
    return tf.multiply(X, w)
w = tf.Variable(3.0, name="weights") 
y_model =  tf.multiply(X, w)
cost = tf.square(Y-y_model) 
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
train_op = tf.train.AdamOptimizer(0.1).minimize(cost) 
sess = tf.Session() 
init = tf.global_variables_initializer() 
sess.run(init) 
for epoch in range(training_epochs): 
    for (x, y) in zip(x_train, y_list): 
        _, loss_val= sess.run([train_op,cost], feed_dict={X: x, Y: y}) 
w_val = sess.run(w) 
print('w=',w_val)
print('loss',loss_val) #single pair loss
sess.close() 
plt.scatter(x_train, y_list) 
y_learned = x_train*w_val 
plt.plot(x_train, y_learned, 'r') 
plt.show()
