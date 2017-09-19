import tensorflow as tf
import numpy as np
with tf.Session() as sess:
    print(tf.random_uniform([4, 10],maxval=10, dtype=tf.int32).eval())
    dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset1.output_types) # ==> "tf.float32"
    print(dataset1.output_shapes) #(10,) #each element shape in dataset
    #tuple () :ordered 
    dataset2 = tf.contrib.data.Dataset.from_tensor_slices((tf.random_uniform([4]),
        tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
    #dictionary {} :string and tensor, add element name
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        {"a": tf.random_uniform([4]),
         "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
    print(dataset.output_types) # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset.output_shapes) # ==> "{'a': (), 'b': (100,)}"
    
    print(dataset2.output_types) # ==> "(tf.float32, tf.int32)"
    print(dataset2.output_shapes) # ==> "((), (100,))"
    dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.output_types) # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset3.output_shapes) # ==> "(10, ((), (100,)))"

    dataset1 = dataset1.map(lambda x: ...)
