import nltk

nltk.download('wordnet')
import tensorflow.keras as keras
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation

import numpy as np
from tensorflow.keras.datasets import cifar10
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import normalize
from cifarvgg.cifar10vgg import cifar10vgg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar100
import os

import configparser
from losses import WeightedCC

import utils

def resnet_layer(
        inputs,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=100):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR100 has 100)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        y
    )

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def lr_schedule_train(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-2
    print("Learning rate: ", lr)
    return lr


def lr_schedule_retrain(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-8
    if epoch > 180:
        lr *= 0.5e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-2
    print("Learning rate: ", lr)
    return lr


def get_callbacks(lr_schedule, save_file):
    checkpoint = ModelCheckpoint(filepath=save_file, monitor="val_acc", verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    return [checkpoint, lr_reducer, lr_scheduler]

def train_model():
    epochs = 200

    input_shape = x_train.shape[1:]
    depth = 1 * 9 + 2
    model = resnet_v2(input_shape=input_shape, depth=depth)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.001),
        metrics=["accuracy"],
    )

    callbacks = get_callbacks(lr_schedule_train, FILEPATH)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
    )

    # Save the retrained model
    model.save(FILEPATH)
    print("Saved trained model at %s " % FILEPATH)

    # Accuracy and Confusion-matrix based evaluations
    utils.evaluate(
        model, x_test, y_test, image_file_name=IMAGE_FILE,
    )

    return model

def retrain_model():
    epochs = 80

    input_shape = x_train.shape[1:]
    depth = 3 * 9 + 2
    model = resnet_v2(input_shape=input_shape, depth=depth)
    model.load_weights(FILEPATH)

    callbacks = get_callbacks(lr_schedule_retrain, RETRAINED_FILEPATH)

    weight_matrix = np.load(WEIGHTED_LOSS_FILE)
    wcc = WeightedCC(weights=weight_matrix)
    explicable_loss = wcc.get_loss()

    model.compile(
        loss=explicable_loss,
        optimizer=Adam(lr_schedule_retrain(0)),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
    )

    # Save the retrained model
    model.save(RETRAINED_FILEPATH)
    print("Saved trained model at %s " % RETRAINED_FILEPATH)

    # Accuracy and Confusion-matrix based evaluations
    utils.evaluate(
        model, x_test, y_test, image_file_name=RETRAINED_IMAGE_FILE,
    )

    return model


if __name__ == '__main__':
    batch_size = 128  # orig paper trained all networks with batch_size=128

    FILEPATH = "./saved_models/cifar100_ResNet29v2_base.200.h5"
    RETRAINED_FILEPATH = "./saved_models/cifar100_ResNet29v2_model_WLF_WN_200.h5"

    c100_wn_cmat_fname = "confusion_matrices/cifar100_EKL_cmat.npy"

    IMAGE_FILE = "images/cmat_c100_resnetv2.png"
    RETRAINED_IMAGE_FILE = "images/cmat_c100_wn_wlf_resnetv2.png"

    WEIGHTED_LOSS_FILE = c100_wn_cmat_fname

    ####################### LOAD DATA ###########################
    #############################################################

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    ####################### GET CMAT ############################
    #############################################################

    try:
        c100_wn_cmat = np.load(c100_wn_cmat_fname)
    except:
        print("[+]: Getting Cifar100 confusion matrix from WordNet")
        c100_wn_cmat = utils.get_cifarX_EKL_cmat("cifar100")
        c100_wn_cmat = normalize(c100_wn_cmat, "l1")
        np.save(c100_wn_cmat_fname, c100_wn_cmat)

    ####################### MODEL RETRAIN #######################
    #############################################################

    # original_model = train_model()
    retrained_model = retrain_model()

    # for i, model in enumerate([original_model, retrained_model]):
    #     predicted_x = model.predict(x_test)
    #     residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)
    #     loss = sum(residuals) / len(residuals)
    #
    #     name = "Original" if i == 0 else "Retrained"
    #     print("[+]: {} accuracy: {}".format(name, 1 - loss))


