import os

import numpy as np
import h5py
#import pandas
from collections import namedtuple
import random
import tensorflow  as tf
# read single patient h5 file

class read_h5:
    'return a dict which keys from STS_002/3/5/12/21/23/31 , values is a namedtuple()'
    def __init__(self, filepath):
        #'lab_petct_vox_5.00mm.h5'
        #protected , _ just a convention
        self._path = os.path.join('E:\Dataset\segmenting-soft-tissue-sarcomas', filepath)
        self.pdata = h5py.File(self._path, 'r')
        self.patient_id = []
        self.data_dict = {}

    def get_patient_id(self):
        for id in self.pdata['ct_data']: # ['ct_data'] ['pet_data'] ['label_data']
            self.patient_id.append(id)
        return self.patient_id

    def get_data_dict(self):
        pdata = namedtuple('PatientData', ['ct', 'pet', 'label', 'valid'])
        #print(len(self.patient_id))
        for id in self.patient_id:
            try:
                ct_data = self.pdata['ct_data'][id]
                pet_data = self.pdata['pet_data'][id]
                label_data = self.pdata['label_data'][id]
                #print('success')
            except KeyError as ke:
                #print('false')
                single = pdata(None, None, None, False)
            single = pdata(np.array(ct_data), np.array(pet_data), np.array(label_data), True)
            self.data_dict[id] = single
        return self.data_dict

def get_numpy_array(start_index, end_index):
    reader = read_h5('lab_petct_vox_5.00mm.h5')
    # 2,3,5,12,21,23,31
    ids = reader.get_patient_id()
    data_dict = reader.get_data_dict()

    pet_volume = np.empty(shape=(0, 100, 100))
    label_volume = np.empty(shape=(0, 100, 100))
    for id in ids[start_index : end_index]:
        pet_volume = np.concatenate((pet_volume, data_dict[id].pet), axis=0)
        label_volume = np.concatenate((label_volume, data_dict[id].label), axis=0)

    return   pet_volume, label_volume

class pet_provider():
    channels =1
    n_class =2
    #sess = tf.Session()
    def __init__(self, mode):
        if mode == 'train':
            self.pet_volume, self.label_volume = get_numpy_array(0,5)
        if mode == 'dev':
            self.pet_volume, self.label_volume = get_numpy_array(5,6)
        if mode == 'test':
            self.pet_volume, self.label_volume = get_numpy_array(6,7)

    def __call__(self, n):
        X = np.zeros((n))#np.zeros((n, 100, 1random.choice00, self.channels))
        Y =  np.zeros((n))#np.zeros((n, 100, 100, 1))

        test_x = [1, 2, 3]
        test_y = [11, 22, 33]
        for i in range(n):
            X[i] = random.choice(test_x)
            Y[i] = random.choice(test_y)
        return X, Y

    def make_data(self, repeat = 5, batch_size = 1):
        self.data = tf.data.Dataset.from_tensor_slices(
            (tf.expand_dims(self.pet_volume, -1),
             tf.one_hot(tf.to_int32(self.label_volume), depth=2, axis=-1)))
        # error map only return y
        # self.data = self.data.map(lambda x,y : tf.one_hot(tf.to_int32(y), depth= 2, axis=-1))
        self.data= self.data.shuffle(buffer_size= 903).batch(batch_size).repeat(repeat)

        return self.data

    def iter_init(self,sess):
        self.iterator = self.data.make_initializable_iterator()
        sess.run(self.iterator.initializer)
        self.next_element = self.iterator.get_next()

    def get_next(self,sess):
        #iterator = self.data.make_initializable_iterator()
        #sess.run(iterator.initializer)
        #next_element = self.iterator.get_next()
        next_element = sess.run(self.next_element)
        return next_element


if __name__ == "__main__":
    pet, label = get_numpy_array(0,5)
    print(pet.shape)
    print(label.shape)
    lab_index = []
    for z in  range(label.shape[0]):
        if label[z].sum()>0:
            lab_index.append((z))
    print(lab_index)
    print(len(lab_index)) #125 ：778
    provider = pet_provider("train")
    X, Y = provider(2)
    print(X)
    print(Y)
    data = provider.make_data()
    print(data.output_shapes)
    sess = tf.Session()
    batch_x ,batch_y = provider.get_next(sess)
    print((batch_y.shape))




