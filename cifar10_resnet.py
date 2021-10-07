import nltk

nltk.download('wordnet')

from resnetv2 import *
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score

from losses import WeightedCC
import utils


def get_callbacks(WEIGHTS_FILE):
    checkpoint = ModelCheckpoint(filepath=WEIGHTS_FILE, monitor="val_accuracy", verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    return [checkpoint, lr_reducer]


def retrain_model(model, cmat_name, cmat_path):
    callbacks = get_callbacks(RETRAINED_WEIGHTS)

    weight_matrix = np.load(cmat_path)
    wcc = WeightedCC(weights=weight_matrix)
    explicable_loss = wcc.get_loss()
    model.compile(
        loss=explicable_loss,
        optimizer=Adam(1e-7),
        metrics=["accuracy"],  # TODO: Can save best model only with val_acc available
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
    model.save(RETRAINED_WEIGHTS)
    print("Saved trained model at %s " % RETRAINED_WEIGHTS)

    y_pred = model.predict(x_test)
    residuals = np.argmax(y_pred, 1) != np.argmax(y_test, 1)

    loss = sum(residuals) / len(residuals)
    print(f"Cifar10 WLF ResNet {cmat_name} Accuracy: {1 - loss}")

if __name__ == '__main__':
    batch_size = 128  # orig paper trained all networks with batch_size=128
    epochs = 30

    dataset = "cifar10"
    cmat_name = "EKL"
    cmat_path = f"confusion_matrices/cifar10_{cmat_name}_cmat.npy"

    RETRAIN, EVALUATE = False, True
    RETRAINED_WEIGHTS = "saved_models/cifar10_resnetv2_{}_new.h5".format(cmat_name)
    CCE_WEIGHTS = "./saved_models/cifar10_ResNet29v2_base.200.h5"

    ####################### LOAD DATA ###########################
    #############################################################

    (x_train, y_train), (x_test, y_test) = utils.load_cifarX(dataset)
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    ####################### GET CMAT ############################
    #############################################################

    print("[+]: Getting Cifar10 confusion matrix")
    c10_wn_cmat = np.load(cmat_path)

    ####################### MODEL RETRAIN #######################
    #############################################################

    input_shape = x_train.shape[1:]
    model = resnet_v2(input_shape=input_shape, depth=3 * 9 + 2)

    if RETRAIN:
        model.load_weights(CCE_WEIGHTS)
        print(f"Accuracy for CCE Resnetv2 {accuracy_score(np.argmax(model.predict(x_test), 1), np.argmax(y_test, 1))}")
        print("[+]: Retraining...")
        retrain_model(model, cmat_name, cmat_path)

    #############################################################
    #############################################################

    if EVALUATE:
        IHL_cmat = np.load("confusion_matrices/cifar10_IHL_cmat.npy")
        np.fill_diagonal(IHL_cmat, 0)
        IHL_expl_labels = np.argmax(IHL_cmat, 1)
        y_test = np.argmax(y_test, 1)

        mapping = {}
        for k, v in enumerate(IHL_expl_labels):
            mapping[k] = v

        for loss_type in ["base.200", "IHL", "CHL", "EKL"]:
            print(f"Evaluating {loss_type}\n--")
            model.load_weights("./saved_models/cifar10_ResNet29v2_{}.h5".format(loss_type))
            y_pred = np.argmax(model.predict(x_test), 1)

            residuals = y_pred != y_test
            y_pred_miscl, y_test_miscl = y_pred[residuals], y_test[residuals]
            # Gets the labels predicted by IHL assuming is our golden model
            y_IHL_golden = np.vectorize(mapping.get)(y_test_miscl)
            acc = accuracy_score(y_IHL_golden, y_pred_miscl)
            print(f"Pred accuracy: {1 - sum(residuals) / len(residuals)} \nAccuracy wrt IHL: {acc}")


##################################################
##################################################
# except:
# c10_wn_cmat = utils.get_cifarX_EKL_cmat("cifar10")
# c10_wn_cmat = normalize(c10_wn_cmat, "l1")
# np.save(c10_wn_cmat_fname, c10_wn_cmat)

# Accuracy and Confusion-matrix based evaluations
# utils.evaluate(
#     model, x_test, y_test, image_file_name=IMAGE_FILE,
# )
