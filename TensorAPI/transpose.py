import tensorflow as tf

# shape(2,2,3)
x = tf.constant([[[ 1,  2,  3],
                  [ 4,  5,  6]],
                 [[ 7,  8,  9],
                  [10, 11, 12]]])
                  
# trans axis 0 and 1  ，x[i,j,k]= x[j,i,k]              
y = tf.transpose(x,perm=[1,0,2])

'''
>>>
array([[[ 1,  2,  3],
        [ 7,  8,  9]],
       [[ 4,  5,  6],
        [10, 11, 12]]])
'''
