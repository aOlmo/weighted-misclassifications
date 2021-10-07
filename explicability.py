"""
This bulk of this code originates from Keras' Resnet for CIFAR 10: https://keras.io/examples/cifar10_resnet/
We modify the loss function and the evaluation metric depending on the classifier to be trained.
"""

from __future__ import print_function
import os

import pandas as pd
import torch
import pickle
import numpy as np
import utils
import torchvision
import tensorflow as tf
import tensorflow.keras

from Imagenet import Imagenet
from resnetv2 import *
from losses import WeightedCC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from torchvision import datasets, transforms
from cifarvgg.cifar100vgg import cifar100vgg
from tensorflow.keras.utils import multi_gpu_model
from utils import get_confusion_matrix, plot_cmat_basic, get_ImageNet_EKL_cmat
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


def get_loss(model, weighted_loss_file, x_test, y_test):
    batch_size = 128  # orig paper trained all networks with batch_size=128
    # Decide which loss function to use.
    if "base" not in weighted_loss_file:
        weight_matrix = np.load(weighted_loss_file)
        wcc = WeightedCC(weights=weight_matrix)
        explicable_loss = wcc.get_loss()
        model.compile(
            loss=explicable_loss,
            optimizer=Adam(learning_rate=lr_schedule(0)),
            metrics=["accuracy"],
        )

    else:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule(0)),
            metrics=["accuracy"],
        )

    print('\n# Evaluate on test data')
    return model.evaluate(x_test, y_test, batch_size=batch_size)

def get_data_from_loader(loader):
    import torch
    f_imgs, f_labels = [], []

    for images, labels in loader:
        f_imgs.append(images)
        f_labels.append(labels)

    return torch.cat(f_imgs, 0), torch.cat(f_labels, 0)

def get_y_test_pred_models(model_files, names):
    for model_file, model_name in zip(model_files, names):
        filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), model_file)
        model.load_weights(filepath)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule(0)),
            metrics=["accuracy"],
        )

        y_pred = model.predict(x_test)
        np.save("saved_models/y_te_pred_c10_{}".format(model_name), y_pred)
        print(model.evaluate(x_test, y_test))

def get_losses_results():
    # Get the explicability table from the paper
    for model_file, model_name in zip(model_files, names):
        filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), model_file)
        model.load_weights(filepath)
        # model = cifar100vgg(train=False, weights_file='{}'.format(filepath)).model
        for cmat_fname, cmat_name in zip(cmat_files, names):
            cmat_file = os.path.join("confusion_matrices", cmat_fname)
            loss, acc = get_loss(model, cmat_file, x_test, y_test)
            print("\n\n[+]: {}-L{} -- Loss: {} Acc: {}".format(model_name, cmat_name, loss, acc))
            print("============================================== ")

def load_cifar10(dataset):
    # Cifar loading of data
    if "cifar100" in dataset:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif "cifar10" in dataset:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    # y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    # y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, x_test), (y_train, y_test), x_train_mean

def get_avg_cmat():
    cmat = np.zeros((10, 10))
    for cmat_name in cmat_files:
        aux_cmat = np.load("confusion_matrices/" + cmat_name)
        if "CHL" in cmat_name: aux_cmat = normalize(aux_cmat, norm="l1")
        # print(np.sum(aux_cmat, 1))
        cmat += aux_cmat
    cmat /= 3
    return cmat

def plot_all_cmats(models_data):
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cmat_base = pd.DataFrame(np.array(models_data["base"]["cmat_miscl"]), columns=cifar10_classes, index=cifar10_classes)
    title = f"{dataset} | Base Misclassifications | Acc {round(accuracy_score(y_test, models_data['base']['y_pred'] ), 2)}"
    plot_cmat_basic(cmat_base, title)
    for model_name in names[1:]:
        title = f"{dataset} | {model_name} Misclassifications | Acc {round(accuracy_score(y_test, models_data[model_name]['y_pred']), 2)}"
        cmat = pd.DataFrame(models_data[model_name]["cmat_miscl"], columns=cifar10_classes, index=cifar10_classes)
        plot_cmat_basic(cmat, title)

        cmat_miscl = np.array(models_data[model_name]["cmat_miscl"])
        cmat_miscl = pd.DataFrame(cmat_miscl, columns=cifar10_classes, index=cifar10_classes)
        title = f"{dataset} | {model_name}-Base misclassification overlap"
        plot_cmat_basic(cmat_miscl-np.array(models_data['base']["cmat_miscl"]), title, cmap=None, vmax=None, vmin=0)

def get_models_data(model, models_data_file, models_data, num_classes=10):
    if os.path.exists(models_data_file):
        models_data = pickle.load(open(models_data_file, 'rb'))
    else:
        for model_file, model_name in zip(model_files, names):
            model.load_weights(os.path.join(os.path.join(os.getcwd(), "saved_models"), model_file))
            y_pred = np.argmax(model.predict(x_test), axis=1)
            y_miscl = (y_pred - y_test) != 0

            models_data[model_name]["y_pred"] = y_pred
            models_data[model_name]["y_wrong"] = np.argwhere(y_miscl == True)

            models_data[model_name]["cmat"] = get_confusion_matrix(y_pred, y_test, n_classes=num_classes)
            models_data[model_name]["cmat_miscl"] = get_confusion_matrix(y_pred[y_miscl], y_test[y_miscl], n_classes=num_classes)
            print(accuracy_score(y_pred, y_test))
        pickle.dump(models_data, open(models_data_file, 'wb'))
    return models_data

def compute_models_hard_soft_scores(model, models_data_file):
    n = len(names)
    models_data = {}
    for x in names:
        models_data[x] = {}

    hard_sim = np.array([0] * n).astype("float32")
    soft_sim = np.array([0] * n).astype("float32")
    soft_sim_prev = np.array([0] * n).astype("float32")

    models_data = get_models_data(model, models_data_file, models_data, num_classes)
    # plot_all_cmats(models_data)

    final_set = set(range(len(x_test)))
    for model_name in names:
        aux_set = set(models_data[model_name]["y_wrong"].flatten())
        final_set = aux_set.intersection(final_set)

    print(f"All classifiers misclassification image count: {len(final_set)}")
    for i in final_set:
        aux_sims, max_sim = [], 0
        for ii, model_name in enumerate(names):
            y_pred = models_data[model_name]["y_pred"]
            sim = cmat[y_test[i], y_pred[i]]
            aux_sims.append(sim)
            max_sim = max(sim, max_sim)

        max_sims = (aux_sims == max_sim)
        # Add max/n where n is the number of same max values in the vector
        aux_soft_sims = [1 / sum(max_sims)] * len(max_sims)
        # aux_soft_prev_sims = aux_sims / sum(max_sims)

        if sum(max_sims) == 1:
            hard_sim[max_sims] += 1

        for i, (ss, flag) in enumerate(zip(aux_soft_sims, max_sims)):
            soft_sim[i] += ss if flag else 0

    return hard_sim, soft_sim

def load_cifar10_imagenet(x_train_mean):
    from skimage.transform import resize
    imgnet = Imagenet()
    batch = 1300

    classes = [x.split(",")[2].strip() for x in list(open("misc/cifar10_to_imagenet_classnames", "r"))]
    x_test, y_test = [], []
    for label, classname in enumerate(classes):
        x, _ = imgnet.get_specific_class_X_Y(classname, batch)
        x, y = resize(x, (len(x), 32, 32, 3)), np.array([label]*len(x))
        x -= x_train_mean
        x_test.append(x)
        y_test.append(y)

    return np.vstack(x_test), np.hstack(y_test)

def load_imagenet(n=10000):
    imgnet = Imagenet()
    x_test, y_test = imgnet.get_X_Y_ImageNet("val", n=n)

    return x_test, y_test

# =================================================================
# =================================================================

# All trained models:
# https://drive.google.com/drive/folders/1NfTMK6JFdPAZTSE5x4Gtdq4YZCFDvhTq?usp=sharing
if __name__ == '__main__':
    ###############################################################################
    #                         Default values (Cifar-10)                           #
    ###############################################################################
    dataset = "cifar10_extra"  # cifar100, cifar100_extra, cifar10, cifar10_c, cifar10_1, cifar10_extra, imagenet, imagenet_cifar

    names = ["base", "IHL", "CHL", "EKL"]
    model_files = ["cifar10_ResNet29v2_base.200.h5", "cifar10_ResNet29v2_IHL.h5", "cifar10_ResNet29v2_CHL.h5", "cifar10_ResNet29v2_EKL.h5"]
    cmat_files = ["cifar10_IHL_cmat_prev.npy", "cifar10_CHL_cmat_prev.npy", "cifar10_EKL_cmat_prev.npy"]
    input_shape = (32, 32, 3) if "cifar" in dataset else (299, 299, 3)

    models_data_file = "misc/models_data_{}.pkl"
    model = resnet_v2(input_shape=input_shape, depth=3 * 9 + 2)
    cmat = get_avg_cmat()

    ###############################################################################
    #                               Get Data                                      #
    ###############################################################################
    (x_train, x_test), (y_train, y_test), x_train_mean = load_cifar10("cifar10")
    if "cifar10_" in dataset or "cinic" in dataset:
        x_test, y_test = utils.load_cifarX(dataset)
        x_test -= x_train_mean
    elif dataset == "imagenet_cifar":
        x_test, y_test = load_cifar10_imagenet(x_train_mean)
    elif dataset == "imagenet":
        model_files = ["ImageNet_ResNetv2_base.h5", "ImageNet1000_ResNetv2_EKL.h5"]
        names = ["base", "EKL"]
        x_test, y_test = load_imagenet()
        model = InceptionResNetV2()
        model = multi_gpu_model(model, 4) if 4 > 1 else model
        cmat = get_ImageNet_EKL_cmat()

    elif dataset == "cifar10":
        y_test = y_test.flatten()
    elif "cifar100" in dataset:
        model_files = ["cifar100_cifarvgg100_base.h5", "cifar100_cifarvgg100_EKL.h5"]
        names = ["base", "EKL"]

        c100obj = cifar100vgg(train=False)
        model = c100obj.model

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        if "cifar100_extra" == dataset:
            x_test, y_test = utils.load_cifarX(dataset)
            x_test *= 255.

        x_train, x_test = c100obj.normalize(x_train, x_test)
        y_test = y_test.flatten()

        cmat = np.load("./confusion_matrices/c100_cmat_EKL_l2.npy")

    num_classes = len(cmat)

    ###############################################################################
    #                           Compute Hard/Soft scores                          #
    ###############################################################################
    hard_sim, soft_sim = 0, 0
    if "cifar10_c" in dataset:
        types = os.listdir("/data/alberto/datasets/CIFAR-10-C/")
        types.remove("labels.npy")
        data_HS, data_SS = pd.DataFrame(columns=names, index=types), pd.DataFrame(columns=names, index=types)

        x_test_orig = x_test.reshape((len(types), -1, 32, 32, 3))
        y_test_orig = y_test.reshape((len(types), -1))
        for i, type in enumerate(types):
            x_test, y_test = x_test_orig[i], y_test_orig[i]
            models_data_file = "misc/models_data_{}.pkl".format(type)
            (data_HS.loc[type], data_SS.loc[type]) = compute_models_hard_soft_scores(model, models_data_file)

        print(names)
        print(np.average(data_HS, axis=0))
        print(np.average(data_SS, axis=0))

    else:
        hard_sim, soft_sim = compute_models_hard_soft_scores(model, models_data_file.format(dataset))

        print(names)
        print("Hard sim: ", np.round(hard_sim, 4))
        print("Soft sim: ", np.round(soft_sim, 4))

    if dataset == "cifar10" or dataset == "cifar10_extra":

        l = ["IHL", "CHL", "EKL"]
        losses = pd.DataFrame(columns=l, index=names)
        accuracies = pd.DataFrame(columns=l, index=names)
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

        # Get the explicability table from the paper
        for model_file, model_name in zip(model_files, names):
            filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), model_file)
            model.load_weights(filepath)
            # model = cifar100vgg(train=False, weights_file='{}'.format(filepath)).model
            for cmat_fname, cmat_name, lossname in zip(cmat_files, names, l):
                    cmat_file = os.path.join("confusion_matrices", cmat_fname)
                    loss, acc = get_loss(model, cmat_file, x_test, y_test)
                    losses.loc[model_name][lossname] = loss
                    accuracies.loc[model_name][lossname] = acc
                    # print("\n\n[+]: {}-L{} -- Loss: {} Acc: {}".format(model_name, cmat_name, loss, acc))
                    # print("============================================== ")

        print(losses)
        print(accuracies)