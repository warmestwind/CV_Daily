#https://ai.intel.com/biomedical-image-segmentation-u-net/
from tf_unet.unet_medi import Unet, Trainer
import tensorflow as tf
import numpy as np
from  tf_unet.pet_provider import pet_provider

# data_flow
provider_train = pet_provider("train")

# net
net = Unet(channels=1, n_class=2, layers = 3, )
# train
trainer = Trainer(net, batch_size =1, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(provider_train,"./unet_trained", training_iters=3, epochs=5, display_step=3)


# data_flow
provider_pred = pet_provider("test")
# prediction
prediction = net.predict("./unet_trained/model.ckpt", provider_pred)



# below is test model input out size
#sess =tf.Session()
#sess.run(tf.global_variables_initializer())


# print(sess.run(tf.shape(net.logits),feed_dict={net.x:np.ones((1,572,572,3)),
#                                              #net.y:np.ones((1,,,2)),
#                                              net.keep_prob:1.0}))

#sess.close()

