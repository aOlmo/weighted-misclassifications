import nltk
nltk.download('wordnet')

from resnetv2 import *
import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10

import configparser
from losses import WeightedCC
import utils

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
        lr *= 0.5e-4
    elif epoch > 160:
        lr *= 1e-4
    elif epoch > 120:
        lr *= 1e-1
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr

def get_callbacks(WEIGHTS_FILE):
    checkpoint = ModelCheckpoint(
        filepath=WEIGHTS_FILE, monitor="val_acc", verbose=1, save_best_only=True
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(
        factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
    )

    return [checkpoint, lr_reducer, lr_scheduler]

def retrain_model(model):
    RETRAINED_WEIGHTS = "saved_models/cifar10_resnetv2_wlf_wordnet_200.h5"

    weight_matrix = np.load(WEIGHTED_LOSS_FILE)
    wcc = WeightedCC(weights=weight_matrix)
    explicable_loss = wcc.get_loss()
    model.compile(
        loss=explicable_loss,
        optimizer=Adam(),
        metrics=["accuracy"],
    )

    callbacks = get_callbacks(RETRAINED_WEIGHTS)

    print("Not using data augmentation.")
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
    # model.save(RETRAINED_WEIGHTS)
    print("Saved trained model at %s " % RETRAINED_WEIGHTS)

    # Accuracy and Confusion-matrix based evaluations
    utils.evaluate(
        model, x_test, y_test, image_file_name=IMAGE_FILE,
    )

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("Cifar10 WLF ResNet Accuracy: ", 1-loss)

if __name__ == '__main__':
    batch_size = 128  # orig paper trained all networks with batch_size=128
    epochs = 100

    FILEPATH = "./saved_models/cifar10_ResNet29v2_base.200.h5"
    c10_wn_cmat_fname = "confusion_matrices/cifar10_EKL_cmat.npy"
    IMAGE_FILE = "images/cmat_c10_resnetv2_wn_new.png"

    WEIGHTED_LOSS_FILE = c10_wn_cmat_fname


    ####################### LOAD DATA ###########################
    #############################################################

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    ####################### GET CMAT ############################
    #############################################################

    # try:
    #     c10_wn_cmat = np.load(c10_wn_cmat_fname)
    # except:
    print("[+]: Getting Cifar10 confusion matrix from WordNet")
    c10_wn_cmat = utils.get_cifarX_EKL_cmat("cifar10")
    c10_wn_cmat = normalize(c10_wn_cmat, "l1")
    np.save(c10_wn_cmat_fname, c10_wn_cmat)

    ####################### MODEL RETRAIN #######################
    #############################################################

    input_shape = x_train.shape[1:]
    # Depth of the initial ResNetv2 model
    depth = 3 * 9 + 2
    model = resnet_v2(input_shape=input_shape, depth=depth)
    model.load_weights(FILEPATH)

    retrain_model(model)

    #############################################################
    #############################################################

    # Accuracy and Confusion-matrix based evaluations
    utils.evaluate(
        model, x_test, y_test, image_file_name=IMAGE_FILE,
    )

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("Accuracy: ", 1-loss)

