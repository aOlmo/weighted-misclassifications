"""
Note to the reviewer: This code uses the ImageNet dataset,
please, use the main download source from http://www.image-net.org/
to get it.
"""

import os
import json
import time
import GPUtil
import argparse
import numpy as np
from ImagenetManualLoad import ImagenetManualLoad


from tqdm import tqdm
from utils import *
from numpy.random import seed
from losses import WeightedCC
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split, ParameterGrid
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from configparser import ConfigParser, ExtendedInterpolation
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input


def compile_WLF_model(model, name, lr=1e-5):
    def lr_schedule(epoch, lr):
        if epoch > 3:
            lr *= 1e-1
        elif epoch > 5:
            lr *= 1e-1
        print("Learning rate: ", lr)
        return lr

    cmat, sc = (get_C10_to_ImageNet_cmat(name, norm=True), 1) if name == "CHL" else (get_ImageNet_EKL_cmat(), 4)
    wcc = WeightedCC(weights=cmat, scaling=sc)
    explicable_loss = wcc.get_loss()
    model.compile(
        loss=explicable_loss,
        optimizer=Adam(learning_rate=lr_schedule(0, lr)),
        metrics=["accuracy"])

    return model


def val_model(model, keras_eval=False, n=50000):
    obj = ImagenetManualLoad()
    x_val, y_val = obj.get_X_Y_ImageNet("val", preprocess=True, n=n)

    if not keras_eval:
        y_pred = np.argmax(model.predict(x_val), 1)

        print("[+]: Calculating validation accuracy...")
        print("[+]: Acc {}".format(accuracy_score(y_val, y_pred)))
    else:
        y_val = to_categorical(y_val, 1000)
        print(model.evaluate(x_val, y_val))


# Note: do not use this training function if evaluation is done on the Validation set
def train_and_save_model(config, model):
    filepath = config["WEIGHTS"]["ekl_train"]

    val_split = float(config["CONST"]["val_split"])
    batch_size = int(config["CONST"]["batch_size"])

    obj = ImagenetManualLoad()
    X, Y = obj.get_X_Y_ImageNet("val", preprocess=True)
    Y = to_categorical(Y, 1000)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=val_split, random_state=SEED, shuffle=False)

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=batch_size, callbacks=[checkpoint])


def train_on_batch(config, model, train_loader):
    savefile = './saved_models/ImageNet_ResNetv2_CHL.h5'
    try:
        for epoch in range(config["CONST"]["epochs"]):
            for i, (x, y) in enumerate(tqdm(train_loader)):
                x = np.rollaxis(x.numpy(), 1, 4)
                # x = preprocess_input(x)
                loss, acc = model.train_on_batch(x, to_categorical(y, 1000))

                if i % 20 == 0:
                    print("[+]: i={}, Loss {} | Acc {}".format(i, loss, acc))

                if i % 10000 == 0 and i != 0:
                    print("[+]: Saving model, loss and acc: [{}, {}]".format(loss, acc))
                    model.save_weights(savefile)

        print("[+]: Saving model, last loss and acc: [{},{}]".format(loss, acc))
        model.save_weights(savefile)

    except KeyboardInterrupt:
        print('[-]: Interrupted, saving model in 3 seconds')
        time.sleep(3)
        print('[+]: Saving model...')
        model.save_weights(savefile)


def train_on_generator(config, model, folder, val_split):
    log_dir = os.path.join(folder, 'logs')
    batch_size = int(config["CONST"]["batch_size"])
    filepath = os.path.join(folder, "EKL-weights-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.2f}.h5")
    os.mkdir(log_dir) if not os.path.isdir(log_dir) else 1

    data_generator = \
        ImageDataGenerator(validation_split=val_split, preprocessing_function=preprocess_input)
    train_generator = \
        data_generator.flow_from_directory(config[DATASET]["train_generator"], target_size=INPUT_SIZE, shuffle=True,
                                           seed=SEED, class_mode='categorical', batch_size=batch_size, subset="training")
    validation_generator = \
        data_generator.flow_from_directory(config[DATASET]["train_generator"],
                                           target_size=INPUT_SIZE, shuffle=True, seed=SEED,
                                           class_mode='categorical', batch_size=batch_size, subset="validation")

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='min',# baseline=ES_BASELINE, 
                                   restore_best_weights=False)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
    logs = TensorBoard(log_dir=log_dir)

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, logs, early_stopping])


def compile_cce(model):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"])

    return model

if __name__ == '__main__':

    config = ImagenetManualLoad().get_config()

    NGPUs = int(config["CONST"]["n_gpu"])
    available_GPUs = GPUtil.getAvailable(order='first', limit=NGPUs, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_GPUs))
    print("[+]:Executing on GPUs: ", os.environ["CUDA_VISIBLE_DEVICES"])
    os.mkdir(config["DIRS"]["curr_model"]) if not os.path.isdir(config["DIRS"]["curr_model"]) else 1

    SEED = int(config["CONST"]["seed"])
    MODELS_F = config["DIRS"]["models"]
    DATASET = config["CONST"]["dataset"]
    EPOCHS = int(config["CONST"]["epochs"])
    ES_BASELINE_VAL_LOSS = 5

    seed(SEED)
    INPUT_SIZE = (299, 299)  # Default input size for ResNetv2-ImageNet

    model = InceptionResNetV2()
    model = multi_gpu_model(model, NGPUs) if NGPUs > 1 else model
    model.save_weights(config["WEIGHTS"]["base"])

    # ops = ["eval_EKL", "eval_EKL_base", "eval_base_EKL", "eval_base"] \
    #     if config["CONST"]["op"] == "test_EKL" else [config["CONST"]["op"]]

    ops = ["eval_EKL_base", "eval_base_EKL"]
    for op in ops:
        print("[+]: Doing op {}".format(op))
        if "train_EKL_grid" == op:
            lr_params = json.loads(config["GRID_SEARCH"]["lr"])
            val_params = json.loads(config["GRID_SEARCH"]["val"])
            param_grid = {"lr": lr_params, "val": val_params}
            grid_list = list(ParameterGrid(param_grid))
            for i, conf in enumerate(grid_list):
                lr, val = conf["lr"], conf["val"]
                folder = "./saved_models/grid_search/{}_lr-{}-val-{}".format(i, lr, val)

                print("------------------ {}/{} ------------------".format(i, len(grid_list)))
                print("[+]: LR: {}, VAL: {}, saving in: ".format(lr, val, folder))
                print("[+]: Loading weights...")
                model.load_weights(config["WEIGHTS"]["base"])

                print("[+]: Compiling model...")
                model = compile_WLF_model(model, "EKL", lr)

                os.makedirs(folder) if not os.path.isdir(folder) else 1
                print("[+]: Training on generator")
                train_on_generator(model, folder, val)
                print("------------------------------------------")

        elif "train_EKL" == op:
            model = compile_WLF_model(model, "EKL")
            folder = folder = "./saved_models/final_FULL/"
            os.makedirs(folder) if not os.path.isdir(folder) else 1
            train_on_generator(config, model, folder, float(config["CONST"]["val_split"]))
       
        elif "eval_EKL" == op:
            model.load_weights(config["WEIGHTS"]["ekl_test"])
            model = compile_WLF_model(model, "EKL")
            val_model(model, keras_eval=True, n=50000)

        elif "eval_base" == op:
            model.load_weights(config["WEIGHTS"]["base"])
            model.compile(
                loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])
            val_model(model, keras_eval=True, n=10000)

        elif "eval_EKL_base" == op:
            model.load_weights(config["WEIGHTS"]["ekl_test"])
            model.compile(
                loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])
            val_model(model, keras_eval=True, n=10000)

        elif "eval_base_EKL":
            model.load_weights(config["WEIGHTS"]["base"])
            model = compile_WLF_model(model, "EKL")
            val_model(model, keras_eval=True, n=10000)

        elif "eval_EKL_hard_soft_scores":
            model.load_weights(config["WEIGHTS"]["ekl_test"])
            model = compile_WLF_model(model, "EKL")