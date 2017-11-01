import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_train = np.linspace(-1, 1, 101) 
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33 
y_train = 2 * x_train 

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



#y_train = 2 * x_train 
#X = tf.placeholder("float") 
X = tf.placeholder("float",[None]) 
#Y = tf.placeholder("float") 
Y = tf.placeholder("float",[None]) 

w = tf.Variable(3.0) 

y_model = X*w # tf.multiply(X, w)
loss = (Y - y_model)**2 #tf.square(Y-y_model) 

train_step=tf.train.AdamOptimizer(0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5):
    #for (x_single, y_single) in zip(x_train, y_train):
        #sess.run(train_step,feed_dict={ X: x_single, Y: y_single})
    _, loss_val = sess.run([train_step, loss],feed_dict={ X:x_train, Y: y_list})
    #print("loss= ",sess.run(loss))
    print("w= ",sess.run(w))

w_val = sess.run(w)
print("loss= ",loss_val) # len(loss_val)= 101
sess.close() 
#plt.scatter(x_train, y_train) 
#plt.show()
plt.scatter(x_train, y_list) 
y_learned = x_train*w_val
plt.plot(x_train, y_learned, 'r') 
plt.show()
