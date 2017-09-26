#import image_preprocess #第二行足够，但是只写import image_preprocess 不行
from image_preprocess import  load_data, image_scaling
import os
import matplotlib.pyplot as plt 
#from train_summary import graph, define_variable
#from train_restore import graph, define_variable
#from test import graph, define_variable
from infer import graph, define_variable
import tensorflow as tf
import numpy as np

def action():
    #define dir
    ROOT_PATH = "F:\SourceCode\Python\Traffic_sign"
    train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Training")
    test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Testing")

    #load data
    #images, labels = load_data(train_data_directory)
    images, labels = load_data(test_data_directory)
    '''
    im=images[14]
    plt.imshow(im)
    plt.show()
    '''
    #images preprocess
    images28=image_scaling(images)

    #train lenet
    graph(images28,labels)

if __name__ == "__main__":
    action()
