import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# generate some fake data

#
def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return (x, y)

data_1 = []
data_2 = []
for i in range(200):
      data_1.append(gauss_2d(1,0.1))

for i in range(200):
      data_2.append(gauss_2d(3,0.1))

label_train = [0]*100+[1]*100
label_dev= [0]*50+[1]*50
label_test= [0]*50+[1]*50


train_data = data_1[0:100]+data_2[0:100]
dev_data =  data_1[100:150]+data_2[100:150]
test_data = data_1[150:]+data_2[150:]

num_epochs = 50
batch_size = 50
dataset_train = tf.data.Dataset.from_tensor_slices((train_data, label_train))
dataset_train = dataset_train.shuffle(100).repeat(num_epochs).batch(batch_size)
dataset_dev= tf.data.Dataset.from_tensor_slices((dev_data, label_dev)).repeat().batch(100)
dataset_test = tf.data.Dataset.from_tensor_slices((test_data, label_test))

print(dataset_train.output_shapes)

# show data
plt.scatter(np.array(dev_data)[:,0], np.array(dev_data)[:,1])
#plt.scatter(np.array(data_2)[:,0], np.array(data_2)[:,1])




def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train :
        iterator = dataset_train.make_one_shot_iterator()
        xs, ys = iterator.get_next()
        #k = 0.5
    else:
        iterator = dataset_dev.make_one_shot_iterator()
        xs, ys = iterator.get_next()
        #k = 1.0
    return xs, ys


# low-level model
def model_low():
    x = tf.placeholder("float", [None, 2])
    # rgb_to_grayscale:return the size of the last dimension of the output is 1, containing the Grayscale value of the pixels.
    y = tf.placeholder(tf.int64, [None])

    h = tf.layers.dense(x, 4)
    y_ = tf.layers.dense(h, 2)

    y_oh = tf.one_hot(y,2)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = y_oh, logits=y_)

    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('Accuracy', accuracy)

    model_path = r"checkpoint_low/"  # ->save()
    saver = tf.train.Saver()  # 保存训练参数
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起

    writer = tf.summary.FileWriter('event_low', sess.graph)

    train_batches_per_epoch = 100 // batch_size  # 每轮多少个batch

    # num of epochs
    for i in range(num_epochs+1):
        #sess.run(training_init_op)
        for step in range(train_batches_per_epoch):
            #img_batch, label_batch = sess.run(next_batch)
            input_, label_ = sess.run(feed_dict(train=True))
            #print("size= ",type(input_[0,0]))
            summary , _, loss, acc = sess.run([merged,train_step,cross_entropy,accuracy] , feed_dict={x : input_, y :label_})
            writer.add_summary(summary, i * train_batches_per_epoch + step)

        if i % 5 == 0:
            save_path = saver.save(sess, model_path, global_step=i)
            print("epoch %d, training accuracy %g" % (i, acc))
            print("epoch %d, training loss %g" % (i, loss))

            # test
            #input_, label_ = sess.run(feed_dict(train=False))
            #print("size= ", (input_.shape))
            summary, acc, loss= sess.run([merged, accuracy, cross_entropy], feed_dict={x : dev_data, y :label_dev})
            print(("epoch %d, dev accuracy %g" % (i, acc)))
            print("epoch %d, dev loss %g" % (i, loss))
            writer.add_summary(summary, i * train_batches_per_epoch + step)

    writer.close()
    sess.close()

model_low()
plt.show()

