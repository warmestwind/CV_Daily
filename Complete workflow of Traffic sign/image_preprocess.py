import os
import skimage
from skimage import transform ,data
import tensorflow as tf
import numpy as np

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:        
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def image_scaling(images):
    images28 = [transform.resize(image, (28, 28)) for image in images]
    return images28

def label_convert(labels):
    #labels preprocess
    labels=tf.one_hot(labels,62)
    #labels=np.asarray(labels)
    print(labels.shape)
    return labels

if __name__ == "__main__":
    print ('This is image_preprocess module')
    
