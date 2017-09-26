import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 28,28,3])
y = tf.placeholder(tf.int64, [None])  


def define_variable(shape,name):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name)

def graph(images,labels):

    images_a=np.array(images)
    print(images_a.shape)
    #labels_a=np.array(labels)
    #print(labels.shape)

    # 784x1 -> 28x28
    input_image = tf.reshape(x, [-1, 28, 28, 3]) #-1 推测batch,28,28 height,width, 1 channel
    tf.summary.image('input', input_image, 10) 

    W_conv1 = define_variable([5, 5, 3, 6], "W_conv1")  # 5 5 height,width ， 1当前层的深度即2维图像深度是1，  6输出层的深度，6个不同的特征图
    tf.summary.histogram('W_conv1', W_conv1)

    b_conv1 = define_variable([6], "b_conv1")
    tf.summary.histogram('b_conv1', b_conv1)

    #   X*W  [None, 784] * [784, 10]                                #padding 加白边，SAME保持卷积后尺寸不变
    conv1 = tf.nn.conv2d(input_image, W_conv1, strides=[1, 1, 1, 1],padding='SAME') #SAME表示全0填充 VALID表示不添充， 由于输入数据被缩放维28 28 ，因此这里第一层卷积保持原有大小使用了SAME
    relu1 = tf.nn.relu(conv1 + b_conv1)
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  #stride是一个四维张量，第一维和最后一维是1，中间两位表示x,y上的步长
    #   ksize=[1, 2, 2, 1] 第一维和第四维是1

    W_conv2 = define_variable([5, 5, 6, 16], "W_conv2")
    tf.summary.histogram('W_conv2', W_conv2)
    b_conv2 = define_variable([16], "b_conv2")
    tf.summary.histogram('b_conv2', b_conv2)

    conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1],padding='VALID') 
    relu2 = tf.nn.relu(conv2 + b_conv2) 
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    W_conv3 = define_variable([5, 5, 16, 120], "W_conv3")
    tf.summary.histogram('W_conv3', W_conv3)
    b_conv3 = define_variable([120], "b_conv3")
    tf.summary.histogram('b_conv3', b_conv3)

    conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1],padding='VALID') 
    relu3 = tf.nn.relu(conv3 + b_conv3)

    flat = tf.reshape(relu3,[-1,120]) #[-1,]表示第一维转换成1维  flattens into 1-D ，行向量

    W_fc1 = define_variable([120,84], "W_fc1")
    tf.summary.histogram('W_fc1', W_fc1)
    b_fc1 = define_variable([84], "b_fc1")
    tf.summary.histogram('b_fc1', b_fc1)

    fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

    dropout1 = tf.placeholder("float")
    fc1_dropout = tf.nn.dropout(fc1, dropout1)

    W_fc2 = define_variable([84, 62], "W_fc2")
    tf.summary.histogram('W_fc2', W_fc2)
    b_fc2 = define_variable([62], "b_fc2")
    tf.summary.histogram('b_fc2', b_fc2)

    y_output = tf.nn.softmax(tf.nn.relu(tf.matmul(fc1_dropout, W_fc2) / dropout1 + b_fc2))  # I think we need compensate here
#以上是infer需要复制的
    y0=tf.one_hot(y,62)
    cross_entropy = -tf.reduce_sum(y0*tf.log(y_output))
    train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)
    #loss
    loss=tf.reduce_mean(cross_entropy)
    tf.summary.histogram('loss', loss)

    correct_prediction = tf.equal(tf.argmax(y_output,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('Accuracy',accuracy)

    model_path="checkpoint/variable"
    saver = tf.train.Saver() #保存训练参数

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint("./checkpoint"))

    merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起

    writer = tf.summary.FileWriter('./Traffic_graph_restore', sess.graph)
    #num of epochs
    for i in range(3400,5001): 
      #batch = mnist.train.next_batch(50)
        if i%100 == 0:
            result = sess.run(merged,feed_dict={ x:images_a, y: labels, dropout1: 1.0})
            writer.add_summary( result,i)

            train_accuracy = accuracy.eval(feed_dict={    
                x:images_a, y: labels, dropout1: 1.0})
            save_path = saver.save(sess, model_path, global_step =i)
            print('here')
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: images_a, y: labels, dropout1: 0.5})
   
    writer.close()


if __name__ == "__main__":
    print ('This is train_lenet module')
