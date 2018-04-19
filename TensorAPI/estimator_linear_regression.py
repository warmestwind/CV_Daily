# -*- coding: UTF-8 -*- 
# or  #coding=utf-8 

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# 定义输入数据
X_data = np.linspace(-2, 2, 101)
noise = np.random.normal(0, 0.5, 101)  # 形成正态分布的误差
noise1 = np.random.random_integers(-5, 5, 101)*0.2
Y_data = 5 * X_data + noise + 1


# 输入数据显示
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1, facecolor=(1,0,1))
ax.scatter(X_data, Y_data)
#plt.show()


# 定义输入函数
def input_fn():
    input_data = {'x': X_data}
    features = dict(input_data)
    dataset = tf.data.Dataset.from_tensor_slices((features, Y_data))
    dataset = dataset.repeat(600).batch(101)
    #x={'x':tf.convert_to_tensor(X_data)}
    #return x, Y_data
    return dataset

numeric_feature_column = [tf.feature_column.numeric_column('x')]

# 定义模型配置 
run_config = tf.estimator.RunConfig(model_dir="./checkpoint")

# 定义模型优化器
optimizer = tf.train.GradientDescentOptimizer(0.001)

# 定义模型
model = tf.estimator.LinearRegressor(feature_columns=numeric_feature_column, optimizer= optimizer, config= run_config )

# 训练模型
model.train(input_fn=input_fn, steps=600)

# 评估
def eval_input():
    e_data = {'x': [-2,-1,0,1,2]}
    e_label = [-9, -4, 1, 6, 11]
    dataset = tf.data.Dataset.from_tensor_slices((dict(e_data),e_label))
    dataset = dataset.batch(2)
    return dataset

eval_result = model.evaluate(eval_input, steps=1)
print("loss= ",eval_result['loss'])

# 预测
# def predict_input2():
#     #p_data = [-2,-1,0,1,2]
#     p_data = [-1]
#     x={'x':tf.convert_to_tensor(p_data)}
#     return x

def predict_input():
    #p_data = [-2,-1,0,1,2]
    p_data = {'x': [-2,-1,0,1,2]}
    features = dict(p_data)
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(2)
    return dataset

predictions = model.predict(predict_input)
for p in predictions:
    print(p['predictions'])

