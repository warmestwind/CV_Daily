This folder contain the complete workflow for classification of traffic sign.  
* act.py is equivalent of main.cpp
* image_preprocess.py implement preprocessing of dataset
* train2.py is single layer neural network
* train3.py is lenet5
* use_onehot.py is the use case for one_hot that can be used in cross entropy
* use_dataset.py is the use case for construct own dataset
* train_summary.py include summary in tensorboard, where can see the changes ofvariables 
* train_restore.py used to reuse the model earling trained, or continue train
* infer.py used to infer the class of input by a trained model
