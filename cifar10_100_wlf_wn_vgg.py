import nltk
import tensorflow.keras as keras
import numpy as np
import utils
import tensorflow

from losses import WeightedCC
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import normalize
from cifarvgg.cifar10vgg import cifar10vgg
from cifarvgg.cifar100vgg import cifar100vgg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10, cifar100

from utils import get_cifarX_EKL_cmat

nltk.download('wordnet')

def retrain_vgg100(model):
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    weight_matrix = np.load(WEIGHTED_LOSS_FILE)
    wcc = WeightedCC(weights=weight_matrix)
    explicable_loss = wcc.get_loss()

    # model = multi_gpu_model(model, 6)

    model.compile(
        loss=explicable_loss,
        optimizer=Adam(learning_rate=1e-7),
        metrics=["accuracy"],
    )

    checkpoint = ModelCheckpoint(filepath=FILEPATH, monitor="val_acc", verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer]

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
    )

    # Save the trained model
    model.save(FILEPATH)
    print("Saved trained model at %s " % FILEPATH)

    # Accuracy and Confusion-matrix based evaluations
    utils.evaluate(model, x_test, y_test, image_file_name=IMAGE_FILE)

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)

    loss = sum(residuals) / len(residuals)
    print("the validation 0/1 accuracy is: ", 1 - loss)

# Best EKL configuration:
    # LR: 1e-7 | epochs: 30 | bsize 1024
    # EKL CMat for cifar100 multiplied by 40 and normalized with l2 and NO softmax on WeightsCC
    # Normalize Cifar100 data with cifar100vgg.normalize(x_train, x_test)
if __name__ == '__main__':
    batch_size = 1024  # orig paper trained all networks with batch_size=128
    epochs = 30

    FILEPATH = "./saved_models/cifar100_cifarvgg100_EKL.h5"
    IMAGE_FILE = "images/cmat_c100_EKL_resnetv2.png"

    c100vgg = cifar100vgg(train=False)
    model = c100vgg.model

    ####################### LOAD DATA ###########################
    #############################################################

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_test = c100vgg.normalize(x_train, x_test)

    y_train = tensorflow.keras.utils.to_categorical(y_train, 100)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 100)

    ####################### GET CMAT ############################
    #############################################################
    WEIGHTED_LOSS_FILE = "./confusion_matrices/c100_cmat_EKL_l2.npy"

    # try:
    #     wn_cmat = np.load(WEIGHTED_LOSS_FILE)
    # except:
    print("[+]: Getting Cifar10 confusion matrix from WordNet")
    wn_cmat = get_cifarX_EKL_cmat("cifar100")
    wn_cmat = normalize(wn_cmat, "l1")*100
    np.save(WEIGHTED_LOSS_FILE, wn_cmat)

    ####################### MODEL RETRAIN #######################
    #############################################################

    # eval_base_cifar10vgg_model = False
    # if eval_base_cifar10vgg_model:
    #     print("[+]: Printing base confusion matrix")
    #     utils.evaluate(model, x_test, y_test, image_file_name="images/cmat_c10_base.png",)
    print(wn_cmat)
    retrain_vgg100(model)