#源代码参考:https://github.com/jakeret/tf_unet
#实验数据:https://www.kaggle.com/4quant/soft-tissue-sarcoma
#医学图像Unet:https://ai.intel.com/biomedical-image-segmentation-u-net/
from tf_unet.unet_medi import Unet, Trainer
import tensorflow as tf
import numpy as np
from  tf_unet.pet_provider import pet_provider
import matplotlib.pyplot as plt

# data_flow
provider_train = pet_provider("train")

# # net
net = Unet(channels=1, n_class=2, layers = 3, )
# # train
# trainer = Trainer(net, batch_size =16, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# path = trainer.train(provider_train,"./unet_trained", training_iters=56, epochs=50, display_step=3)


# data_flow
provider_pred = pet_provider("test")

# prediction
prediction , x, y, acc= net.predict("./unet_trained/model.ckpt", provider_pred)
np.save('031unet.npy',prediction[...,1])
print('最大概率 ', prediction[...,1].max())
print('最大标签 ',y[...,1].max())
print(prediction .shape)
print(x.shape)
print(y.shape)
print("acc ", acc)
fig, (ax,ax1,ax2) = plt.subplots(1,3,figsize=(12,7),facecolor='pink')
ax.set_title('x')
ax1.set_title('y')
ax2.set_title('pred')
ax.imshow(x[35,...,0],cmap='gray')
ax1.imshow(y[35,...,0],cmap='gray')
ax2.imshow(prediction[35,...,1],cmap='gray')
plt.show()

#-------------------------------------------
# below is test model input out size
#sess =tf.Session()
#sess.run(tf.global_variables_initializer())


# print(sess.run(tf.shape(net.logits),feed_dict={net.x:np.ones((1,572,572,3)),
#                                              #net.y:np.ones((1,,,2)),
#                                              net.keep_prob:1.0}))

#sess.close()

