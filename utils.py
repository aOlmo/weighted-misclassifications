from __future__ import print_function

import os
import nltk
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torchvision import transforms, datasets
from Imagenet import Imagenet
from tensorflow.keras.datasets import cifar100

np.random.seed(1337)  # for reproducibility

from tqdm import tqdm
from scipy import ndimage
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import normalize


NUM_CLASSES = 10

# --------------------------------------------------------------- #
# -                          Cifar10                            - #
# --------------------------------------------------------------- #
def load_cifarX(dataset):
    if "cifar10_1" in dataset:
        path = "/data/alberto/datasets/CIFAR-10.1/datasets/"
        x = os.path.join(path, "cifar10.1_v6_data.npy")
        y = os.path.join(path, "cifar10.1_v6_labels.npy")
        x_test, y_test = np.load(x).astype("float32")/255, np.load(y)

    elif "cifar10_extra" in dataset or "cifar100_extra" in dataset:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.ImageFolder('/data/alberto/datasets/{}/'.format(dataset.upper()), transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        x_test, y_test = next(iter(train_loader))
        x_test, y_test = x_test.numpy(), y_test.numpy()

        x_test = np.transpose(x_test, (0, 2, 3, 1))

    elif "cifar10_c" in dataset:
        path = "/data/alberto/datasets/CIFAR-10-C/"
        x_test, y_test = [], []

        types = os.listdir(path)
        types.remove("labels.npy")
        for type in types:
            x = os.path.join(path, f"{type}")
            x_test.append(np.load(x).astype("float32")/255)
            y = np.load(os.path.join(path, "labels.npy"))
            y_test.append(y)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test).flatten()

    elif "cinic" in dataset:
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        cinic_directory = '/data/alberto/datasets/CINIC10'

        x_cinic_file = "./tmp/x_cinic.npy"
        y_cinic_file = "./tmp/y_cinic.npy"
        if not os.path.exists(x_cinic_file):
            cinic_size = 90000
            cinic_test = torch.utils.data.DataLoader(
                datasets.ImageFolder(cinic_directory,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()])),
                batch_size=cinic_size, shuffle=True)

            x_test, y_test = next(iter(cinic_test))
            x_test, y_test = x_test.numpy(), y_test.numpy()
            x_test = np.transpose(x_test, (0, 2, 3, 1))

            np.save(x_cinic_file, x_test)
            np.save(y_cinic_file, y_test)
        else:
            x_test = np.load(x_cinic_file)
            y_test = np.load(y_cinic_file)

    elif "cifar100" == dataset:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Normalize data.
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    return x_test, y_test

# --------------------------------------------------------------- #
# -                          ImageNet                           - #
# --------------------------------------------------------------- #
def get_Cifar10_to_ImageNet_classes():
    file = open("./misc/cifar10_to_imagenet_classnames")
    labels = []
    for line in file:
        labels.append(int(line.strip().split(",")[1]))

    return np.array(labels)

def get_ImageNet_CHL_cmat(savefname=False, norm=True):
    name = "CHL"
    norm_unnorm = "normalized" if norm else "unnormalized"
    savefile = savefname if savefname else "./confusion_matrices/ImageNet_{}_cmat_{}.npy".format(name, norm_unnorm)
    C10_cmat_filename = "./confusion_matrices/cifar10_{}_cmat.npy".format(name)

    if not os.path.exists(savefile):
        ImageNet_cmat = np.identity(1000, dtype=float)*4.0
        C10_cmat = np.load(C10_cmat_filename)
        map = get_Cifar10_to_ImageNet_classes()

        for i in range(C10_cmat.shape[0]):
            for j in range(C10_cmat.shape[1]):
                i0, j0 = map[i], map[j]
                ImageNet_cmat[i0, j0] = C10_cmat[i, j]

        if norm:
            ImageNet_cmat = normalize(ImageNet_cmat, "l1")

        print("[+]: Saving ImageNet {} in {}".format(name, savefile))
        np.save(savefile, ImageNet_cmat)
        return ImageNet_cmat
    print("[+]: Loading ImageNet {} from {}".format(name, savefile))
    return np.load(savefile)

def get_ImageNet_EKL_cmat():
    nltk.download('wordnet')
    savefile = "./confusion_matrices/ImageNet_EKL_cmat.npy"

    obj = Imagenet()
    imagenet_dict = obj.load_dict()

    try:
        cmat = np.load(savefile)
        print("[+]: Loaded {}".format(savefile))
        return cmat

    except:
        # Get dictionary and go one by one
        conf_mat = []
        for cl_row in tqdm(imagenet_dict.keys()):
            row = []
            print("[+]: Doing class {}".format(cl_row))
            row_class = wn.synset_from_pos_and_offset(cl_row[0], int(cl_row[1:]))

            for cl_col in imagenet_dict.keys():
                col_class = wn.synset_from_pos_and_offset(cl_col[0], int(cl_col[1:]))
                row.append(row_class.path_similarity(col_class))

            conf_mat.append(row)
        conf_mat = np.array(conf_mat)
        print("[+]: Saving {}".format(savefile))
        np.save(savefile, conf_mat)
        return conf_mat

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

def get_cifarX_EKL_cmat(cifar_dataset):
    import nltk
    from nltk.corpus import wordnet as wn

    nltk.download('wordnet')

    # Load cifarX class names
    cifarX_class_names = "./misc/cifar10_class_names" if cifar_dataset == "cifar10" \
        else "./misc/cifar100_to_wn_classnames"

    with open(cifarX_class_names, "r") as file:
        c10_classes = [line.strip() for line in file]

    conf_mat = []
    for cl_row in c10_classes:
        row = []
        print("[+]: Doing class {}".format(cl_row))
        row_class = wn.synset('{}.n.01'.format(cl_row))

        for cl_col in c10_classes:
            col_class = wn.synset('{}.n.01'.format(cl_col))
            row.append(row_class.path_similarity(col_class))

        conf_mat.append(row)
    conf_mat = np.array(conf_mat)

    return conf_mat

def plot_confusion_matrix(cmat, dataset_name):
    with open("misc/{}_class_names".format(dataset_name), "r") as file:
        classes = [line.strip() for line in file]

    # aux = normalize(cmat, norm='l1', axis=1)

    data = pd.DataFrame(data=cmat, index=classes, columns=classes)

    plt.figure(figsize=(12, 10))
    plt.title("{} Confusion Matrix".format(dataset_name))
    sns.heatmap(data, cmap="YlGnBu", vmax=0.1)
    plt.savefig("images/{}_conf_mat.png".format(dataset_name))
    plt.show()


def plot_cmat(matrix):
    sns.set()
    sns.set_context("paper")

    # This way of color scale grading preserves trends for low and high values
    lower_side = np.linspace(0, 101, 1000)
    higher_side = np.linspace(700, 961, 2000)
    bounds = np.concatenate((lower_side, higher_side))
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", norm=norm)

def plot_cmat_basic(cmat, title="title", cmap="YlGnBu", vmax=1500, vmin=None):
    sns.heatmap(cmat, cmap=cmap, fmt="d", annot=True, annot_kws={"fontsize":8}, vmax=vmax, vmin=vmin).set_title(title)
    plt.show()

def get_confusion_matrix(y_pred, y_test, n_classes=10):
    if len(y_pred.shape) > 1: y_pred = np.argmax(y_pred, axis=0)
    if len(y_test.shape) > 1: y_test = np.argmax(y_test, axis=0)

    n = len(y_test)
    cm = [[0 for i in range(n_classes)] for i in range(n_classes)]
    for i in range(n):
        true = y_test[i]
        pred = y_pred[i]
        cm[true][pred] += 1
    return cm


def get_pred_accu(model, X, Y, ADD_RANDOM_NOISE=False, DENOISE=False):
    if ADD_RANDOM_NOISE:
        X = addRandomNoiseToBatch(X)
        if DENOISE:
            # Also denoise this batch of images
            X = deNoiseBatch(X)

    NUM_CLASSES = Y.shape[-1]
    y_pred = model.predict(X)

    confusion_matrix = get_confusion_matrix(y_pred, Y)
    for i in range(NUM_CLASSES):
        print(confusion_matrix[i])

    return confusion_matrix


def evaluate(model, x_test, y_test, x_adv=None, image_file_name="images/dummy.png"):
    c_matrix = get_pred_accu(model, x_test, y_test)
    plot_cmat(c_matrix)
    plt.savefig(image_file_name)


def addRandomNoise(x):
    x += np.random.normal(0, 1, x.shape)


def addRandomNoiseToBatch(X):
    for x in X:
        addRandomNoise(x)
    return X


def deNoise(x):
    ndimage.gaussian_filter(x, 1)


def deNoiseBatch(X):
    for x in X:
        deNoise(x)
    return X

