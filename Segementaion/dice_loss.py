import numpy as np
import tensorflow as tf

pred = np.ones((1, 3, 3, 3, 2), dtype=np.float32)
pred[..., 0] = 0

gt = np.ones((1, 3, 3, 3, 2), dtype=np.float32)
gt[..., 0] = 0

def dice_loss_1(pred, gt):
    eps = 1e-5
    intersection = tf.reduce_sum(pred * gt)
    union = eps + tf.reduce_sum(pred) + tf.reduce_sum(gt)
    loss = -(2 * intersection / union)  # 取反求最小值
    return loss

def dice_loss_2(pred, input_gt):
    input_gt = tf.one_hot(input_gt, 2)
    print("shape: ", input_gt.shape)
    dice = 0
    eps = 1e-5
    for i in range(2):
        inse = tf.reduce_sum(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
        l = tf.reduce_sum(pred[:, :, :, :, i] * pred[:, :, :, :, i])
        r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
        dice = dice + 2*inse/(l + r + eps)
    return -dice

def dice_loss_3(pred, input_gt):
    input_gt = tf.one_hot(input_gt, 2)
    print("shape: ", input_gt.shape)
    eps = 1e-5
    inse = tf.reduce_sum(pred*input_gt)
    l = tf.reduce_sum(pred)
    r = tf.reduce_sum(input_gt)
    print(inse, l, r)
    dice = 2*inse/(l + r + eps)
    return -dice

tf.enable_eager_execution()
input_gt = np.ones((1, 3, 3, 3), dtype=np.int32)

print(dice_loss_1(pred, gt))
print(dice_loss_2(pred, input_gt))
print(dice_loss_3(pred, input_gt))
