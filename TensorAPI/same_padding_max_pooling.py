import tensorflow as tf  
# two different input ,when  padding = SAME  result is different from VALID
'''
a=tf.constant( 
        [[1.0,2.0,3.0,4.0],  
        [5.0,6.0,7.0,8.0],  
        [8.0,7.0,6.0,5.0],  
        [4.0,3.0,2.0,1.0]]
) 
''' 
a=tf.constant( 
        [[1.0,2.0,3.0,4.0,1.0],  
        [5.0,6.0,7.0,8.0,2.0],  
        [8.0,7.0,6.0,5.0,8.0],  
        [4.0,3.0,2.0,1.0,3.0]]
)  
  
a=tf.reshape(a,[1,4,5,1])  
pooling=tf.nn.max_pool(a,[1,2,2,1],[1,2,2,1],padding='SAME')  
#pooling=tf.nn.max_pool(a,[1,2,2,1],[1,2,2,1],padding='VALID')  
with tf.Session() as sess:  

    print("reslut:")  
    result=sess.run(pooling)  
    print (result)  
