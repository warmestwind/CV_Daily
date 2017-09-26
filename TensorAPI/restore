import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model-1000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    f=open("out.txt","w")
    #Set printing options. 
    np.set_printoptions(threshold=np.inf) 
    #Total number of array elements which trigger summarization rather than full repr
    print(v_,file=f)
    print(v_)
    #if saver contain an op,then can call it by name and index
    #print(tf.get_default_graph().get_tensor_by_name("add:0"))
