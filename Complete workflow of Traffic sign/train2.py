import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()




def define_variable(shape,name):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name)

def graph(images,labels):
    images_a=np.array(images)
    print(images_a.shape)
    labels_a=np.array(labels)
    print(labels_a.shape)

    print("done")

    
     # Placeholders for inputs and labels.
    x = tf.placeholder(tf.float32, [None, 28, 28, 3])
    y = tf.placeholder(tf.int32, [None])

    # 784x1 -> 28x28
    images_flat = tf.contrib.layers.flatten(x) #之前写成了images

    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, 
                                                                    logits = logits))
    train= tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)      
    correct_pred = tf.argmax(logits, 1)
    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                                                      
    model_path="checkpoint/variable"
    saver = tf.train.Saver() #保存训练参数

    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    #num of epochs  
    for i in range(201):
        _, loss_value = sess.run(
            [train, loss], 
            feed_dict={x: images_a, y: labels_a})
        if i % 10 == 0:
            print("Loss: ", loss_value)

    writer = tf.summary.FileWriter('./Traffic_graph', sess.graph)
    writer.close()


if __name__ == "__main__":
    print ('This is train_fully connect module')
