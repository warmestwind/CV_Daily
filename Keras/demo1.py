from keras.models import Sequential
import keras.backend as K

model = Sequential()

from keras.layers import Dense
import keras

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

import numpy as np

x_train = np.random.random((1000, 100))
y_train = np.random.randint(10 , size=1000)
print(y_train.shape)
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(y_train, num_classes = 10)
#print(categorical_labels)

model.fit(x_train, categorical_labels, epochs=5, batch_size=32,verbose= 0)



weight = model.layers[-1].get_weights()
print(len(weight[0]))


#
x_test = np.random.random((1, 100))
# y_test = np.random.random((100, 10))
#
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

classes = model.predict(x_test)
print(np.sum(classes)) #sum pi =1

get_output  = K.function([model.layers[0].input], [model.layers[-1].output])

print(get_output([x_test]))
#classes = model.predict(x_test, batch_size=128)

