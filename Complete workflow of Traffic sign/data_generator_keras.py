#!/usr/bin/env python
#-*- coding: utf-8 -*-
# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import skimage
from skimage import transform ,data, io
import numpy as np

ROOT_PATH = "F:\SourceCode\Python\Traffic_sign"
train_data_directory = os.path.join(ROOT_PATH, "data_argu2\Training")
test_data_directory = os.path.join(ROOT_PATH, "data_argu\Testing")

datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def load_data(data_directory):
    directories = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    imdir=[]
    imdir2=[]
   
    for d in directories:
        #print(d)
        
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        for f in file_names:  
            img = load_img(f)  # this is a PIL image, please replace to your own file path
            x = img_to_array(img)  # this is a Numpy array with shape (3, h, w)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, h, w)
            i = 0
            for batch in datagen.flow(x, 
					                  batch_size=1,
                                      #save_to_dir='F:\\SourceCode\\Python\\Traffic_sign\\data_argu\\temp',
                                      save_to_dir= d,  
                                      save_prefix='', 
                                      save_format='jpg'):
                i += 1
                if i > 1:
                    break  # otherwise the generator would loop indefinitely

    
load_data(train_data_directory)
