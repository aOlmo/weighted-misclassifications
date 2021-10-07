"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
It is very similar to mnist_tutorial_tf.py, which does the same
thing but without a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import numpy as np

import cleverhans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

# from cleverhans.utils_tf import model_eval
from cleverhans.dataset import CIFAR10

from losses import WeightedCC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 2
LEARNING_RATE = 0.001
TRAIN_DIR = "train_dir"
FILENAME = "cifar10.ckpt"
LOAD_MODEL = False

D_Cifar10 = {
    "Base-LIHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_base.200.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_IHL_cmat.npy",
    },
    "Base-LCHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_base.200.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_CHL_cmat.npy",
    },
    "Base-LEKL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_base.200.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_EKL_cmat.npy",
    },

    ###### IHL ######
    "IHL-LIHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_IHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_IHL_cmat.npy",
    },
    "IHL-LCHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_IHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_CHL_cmat.npy",
    },
    "IHL-LEKL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_IHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_EKL_cmat.npy",
    },

    ###### CHL ######
    "CHL-LIHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_CHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_IHL_cmat.npy",
    },
    "CHL-LCHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_CHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_CHL_cmat.npy",
    },
    "CHL-LEKL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_CHL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_EKL_cmat.npy",
    },

    ###### EKL ######
    "EKL-LIHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_EKL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_IHL_cmat.npy",
    },
    "EKL-LCHL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_EKL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_CHL_cmat.npy",
    },
    "EKL-LEKL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/cifar10_ResNet29v2_EKL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/cifar10_EKL_cmat.npy",
    },
}

D_ImageNet1000 = {
    "EKL": {
        "WLF": True,
        "EVAL_MODEL_FNAME": "saved_models/ImageNet1000_ResNetv2_EKL.h5",
        "WEIGHTED_LOSS_FILE": "confusion_matrices/ImageNet_EKL_cmat.npy",
    },
}

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr

# Using code from https://github.com/sailik1991/MTDeep
# --- DeepFool ---
def get_DF_cg(sess, wrap, x):
    attack = cleverhans.attacks.DeepFool(wrap, back="tf", sess=sess)
    # Consider only three class when searching for a successful attack perturbation
    attack_params = {"nb_candidate": 3}
    adv_x = attack.generate(x, **attack_params)
    adv_x = tf.stop_gradient(adv_x)
    return adv_x


# --- Fast Gradient Method ---
def get_FGM_cg(sess, wrap, x):
    attack = cleverhans.attacks.FastGradientMethod(wrap, sess=sess)
    attack_params = {"eps": 0.3, "clip_min": 0.0, "clip_max": 1.0}
    adv_x = attack.generate(x, **attack_params)
    adv_x = tf.stop_gradient(adv_x)
    return adv_x

# --- Projected Gradient Descent ---
def get_PGD_cg(sess, wrap, x, y):
    attack = cleverhans.attacks.ProjectedGradientDescent(wrap, sess=sess)
    attack_params = {"eps": 0.3, "eps_iter": 0.05, "y": y}
    adv_x = attack.generate(x, **attack_params)
    adv_x = tf.stop_gradient(adv_x)
    return adv_x

# --- Random noise ---
def add_random_noise(w, mean=0.0, stddev=1.0):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return w + noise

def get_np_adv_x(sess, x, adv_x, x_base, batch_size):
    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(x_base)) / batch_size))
        X_cur = np.zeros((batch_size,) + x_base.shape[1:], dtype=x_base.dtype)
        np_adv_x = []
        for batch in range(nb_batches):
            start = batch * batch_size
            end = min(len(x_base), start + batch_size)
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = x_base[start:end]

            np_adv_x.append(adv_x.eval(feed_dict={x: X_cur[:cur_batch_size]}))

    np_adv_x = np.vstack(np_adv_x)
    return np_adv_x

# Code adapted from MNIST tutorial in: https://bit.ly/31R0AI0
def cifar10_tutorial(
        train_start=0,
        train_end=60000,
        test_start=0,
        test_end=10000,
        nb_epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        train_dir=TRAIN_DIR,
        filename=FILENAME,
        testing=False,
        label_smoothing=0.1,
):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """
    tf.keras.backend.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if keras.backend.image_data_format() != "channels_last":
        raise NotImplementedError(
            "this tutorial requires keras to be configured to channels_last format"
        )

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    tf.keras.backend.set_session(sess)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    y_test = keras.utils.to_categorical(y_test, 10)

    # Random noise
    # x_test_noise = x_test + np.random.normal(0, 0.2, x_test.shape)

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    for name in D_Cifar10:
        d = D_Cifar10[name]

        weight_matrix = np.load(d["WEIGHTED_LOSS_FILE"])
        wcc = WeightedCC(weights=weight_matrix)
        explicable_loss = wcc.get_loss()
        model = load_model(
            d["EVAL_MODEL_FNAME"],
            custom_objects={"w_categorical_crossentropy": explicable_loss},
        )

        # preds = model(x)
        print("Defined TensorFlow model graph.")

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        # Abstracting creation of computation graph creation for any attack.
        wrap = KerasModelWrapper(model)
        adv_x = get_FGM_cg(sess, wrap, x)
        # adv_x = get_PGD_cg(sess, wrap, x, y)
        # adv_x = get_DF_cg(sess, wrap, x)

        np_adv_x_test = get_np_adv_x(sess, x, adv_x, x_test, batch_size)
        # y_pred = model.predict(np_adv_x_test)

        loss, acc = model.evaluate(np_adv_x_test, y_test)

        print("[{}]: Cifar10 acc on perturbed test data: {} | loss: {}\n".format(name, acc, loss))

    return report


def imagenet_adv():
    from Imagenet import Imagenet
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    from tensorflow.keras.utils import multi_gpu_model
    from sklearn.metrics import accuracy_score

    # Create TF session and set as Keras backend session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)

    def create_adversarial_pattern(input_image, input_label, model):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        input_image = tf.convert_to_tensor(input_image)

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    def run_adv(image, label, model):
        eps = 0.3
        perturbations = create_adversarial_pattern(image, label, model)

        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)

        return adv_x

    tf.keras.backend.set_learning_phase(0)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    N = 32
    chunks = N//32
    obj = Imagenet()
    x_test, y_test = obj.get_X_Y_ImageNet("val", preprocess=True, save_new=False, n=N)
    y_test = keras.utils.to_categorical(y_test, 1000)

    # Random noise
    x_test_noise = x_test + np.random.normal(0, 0.2, x_test.shape)

    imgs = []
    for name in D_ImageNet1000:
        d = D_ImageNet1000[name]

        weight_matrix = np.load(d["WEIGHTED_LOSS_FILE"])
        wcc = WeightedCC(weights=weight_matrix)
        # wcc = WeightedCC(weights=weight_matrix, scaling=4)
        explicable_loss = wcc.get_loss()

        model = InceptionResNetV2()
        model = multi_gpu_model(model, 2)
        model.load_weights(d["EVAL_MODEL_FNAME"])
        model.compile(
            loss=explicable_loss,
            optimizer=Adam(),
            metrics=["accuracy"])

        for chunk in range(chunks):
            s = chunk*32
            e = s+32
            img = run_adv(x_test[s:e], y_test[s:e], model)
            imgs.append(sess.run(img))

        imgs = np.vstack(imgs)

        print("-------------------- Ours ------------------------")
        print(accuracy_score(np.argmax(model.predict(x_test), 1), np.argmax(y_test, 1)))
        print(accuracy_score(np.argmax(model.predict(imgs), 1), np.argmax(y_test, 1)))
        print("Random noise acc: {}".format(
            accuracy_score(np.argmax(model.predict(x_test_noise), 1), np.argmax(y_test, 1))))

        print("-------------------- Theirs ----------------------")
        model = InceptionResNetV2()
        print(accuracy_score(np.argmax(model.predict(x_test), 1), np.argmax(y_test, 1)))
        print(accuracy_score(np.argmax(model.predict(imgs), 1), np.argmax(y_test, 1)))
        print("Random noise acc: {}".format(
            accuracy_score(np.argmax(model.predict(x_test_noise), 1), np.argmax(y_test, 1))))

def main(argv=None):
    from cleverhans_tutorials import check_installation

    check_installation(__file__)
    # cifar10_tutorial(
    #     nb_epochs=FLAGS.nb_epochs,
    #     batch_size=FLAGS.batch_size,
    #     learning_rate=FLAGS.learning_rate,
    #     train_dir=FLAGS.train_dir,
    #     filename=FLAGS.filename,
    # )

    imagenet_adv()

# TODO: Remember to execute this with the proper conda environment: FGSM_ATTACK
if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", NB_EPOCHS, "Number of epochs to train model")
    flags.DEFINE_integer("batch_size", BATCH_SIZE, "Size of training batches")
    flags.DEFINE_float("learning_rate", LEARNING_RATE, "Learning rate for training")
    flags.DEFINE_string("train_dir", TRAIN_DIR, "Directory where to save model.")
    flags.DEFINE_string("filename", FILENAME, "Checkpoint filename.")
    flags.DEFINE_boolean("load_model", LOAD_MODEL, "Load saved model or train.")

    tf.app.run()