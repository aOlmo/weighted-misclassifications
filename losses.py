from __future__ import print_function

from functools import partial

import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from scipy.special import softmax


def weighted_categorical_crossentropy(y_true, y_pred, weights):
    # y_true : Tensor array with one of true labels
    # y_pred : Tensor array with output labels
    # weights : Tensor array with weight of misclassifiaction matrix
    correct_label_index = K.argmax(y_true)
    cli_weights = tf.gather(weights, correct_label_index)
    return K.categorical_crossentropy(cli_weights, y_pred)


class WeightedCC:
    def __init__(self, weights, threshold=0.5, scaling=1):
        weights *= scaling
        weights = softmax(weights, axis=1)
        self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        # print(self.weights)
        # print(tf.gather(self.weights, 3))

    def get_loss(self):
        # Define parameters for training
        nwlf = partial(weighted_categorical_crossentropy, weights=self.weights)
        nwlf.__name__ = 'w_categorical_crossentropy'
        return nwlf


# Test code
# if __name__ == '__main__':
#     CHL = np.load('confusion_matrices/cifar10_CHL_cmat.npy')
#     wcc = WeightedCC(weights=CHL)
#     exp_loss = wcc.get_loss()
#     print(exp_loss)
