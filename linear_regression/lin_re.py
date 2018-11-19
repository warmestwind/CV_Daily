# encoding:utf-8

"""
python实现线性规划
"""


import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 定义输入数据
X_data = np.linspace(-2, 2, 101)
noise = np.random.normal(0, 0.5, 101)  # 形成正态分布的误差
noise1 = np.random.randint(-5, 5, 101)*0.2
y_data = 5 * X_data + noise1 + 1

# 输入数据显示
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_data, y_data)
# plt.show()

# 将数据放入迭代器
train_iter = mx.io.NDArrayIter(X_data, y_data, batch_size=1, shuffle=False, label_name='softmax_label')

# 定义网络与模型
# 定义mx-net变量
X = mx.symbol.Variable('data')
Y = mx.symbol.Variable('softmax_label')

# 定义网络
Y_ = mx.symbol.FullyConnected(data=X, num_hidden=1, name='pre')
loss = mx.symbol.LinearRegressionOutput(data=Y_, label=Y, name='loss')


# 定义模型
model = mx.mod.Module(
    symbol=loss,
    data_names=['data'],
    label_names=['softmax_label']
)
mx.viz.plot_network(loss).view()
# 训练模型
model.fit(train_iter,
          optimizer_params={'learning_rate': 0.005, 'momentum': 0.9},
          num_epoch=50,
          eval_metric='mse',
          batch_end_callback=mx.callback.Speedometer(1, 2))

# 用训练好的模型预测
eval_iter1 = mx.io.NDArrayIter(X_data, y_data, batch_size=1, shuffle=False)
prediction = model.predict(eval_iter1)

# 绘制预测直线图
fig1 = plt.figure()
"""ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(X_data, prediction.asnumpy(), color='red')
ax1.scatter(X_data, y_data)"""

ax2 = fig1.add_subplot(1, 1, 1)
x1, x2, y1, y2 = min(X_data), max(X_data), min(prediction.asnumpy()), max(prediction.asnumpy())
ax2.set_xlim(left=x1, right=x2)
ax2.set_ylim(bottom=y1, top=y2)
line1 = [(x1, y1), (x2, y2)]
(line1_xs, line2_xs) = zip(*line1)
ax2.add_line(Line2D(line1_xs, line2_xs, linewidth=1, color='red'))
ax2.scatter(X_data, y_data)
plt.show()

'''print(type(X_data), type(prediction.asnumpy()))
plt.plot(X_data, prediction.asnumpy())
# lines = ax.plot(X_data, prediction)
plt.show()'''


# 设置测试参数
eval_data = np.array([-2, -1, 0, 1, 2])
eval_label = np.array([-9, -4, 1, 6, 11])
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size=1, shuffle=False)

print(model.predict(eval_iter).asnumpy())

metric = mx.metric.MSE()
model.score(eval_iter, metric)
print(metric.get())
