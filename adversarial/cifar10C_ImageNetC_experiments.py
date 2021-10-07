import os
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import multi_gpu_model
from resnetv2 import *

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Tiny-ImageNet-C and Cifar10C can be found here:
# https://github.com/hendrycks/robustness

def do_cifar10c():
    (x_train, y_train), (_, _) = cifar10.load_data()

    d = {}
    for C10_perturb_file in os.listdir(CIFAR10C_DIR):
        if "labels" in C10_perturb_file: continue
        d[C10_perturb_file] = []

    types = ["base", "IHL", "CHL", "EKL"]
    for type in types:
        input_shape = x_train.shape[1:]
        model = resnet_v2(input_shape=input_shape, depth=3 * 9 + 2)
        model.load_weights(FILEPATHS[type])

        y_test = np.load(os.path.join(CIFAR10C_DIR, "labels.npy"))
        for C10_perturb_file in os.listdir(CIFAR10C_DIR):
            if "labels" in C10_perturb_file: continue
            x_test = np.load(os.path.join(CIFAR10C_DIR, C10_perturb_file))

            predicted_x = model.predict(x_test)

            correct = np.argmax(predicted_x, 1) == y_test
            accuracy = sum(correct) / len(correct)

            d[C10_perturb_file].append(accuracy)

            print("[{}]: {} accuracy: {}".format(type, C10_perturb_file, accuracy))

    for C10_perturb_file in os.listdir(CIFAR10C_DIR):
        if "labels" in C10_perturb_file: continue
        print("{}: {}".format(C10_perturb_file, types[int(np.argmax(np.array(d[C10_perturb_file])))]))
        print("{}: {}".format(C10_perturb_file, types[int(np.argmax(np.array(d[C10_perturb_file][1:])) + 1)]))


if __name__ == '__main__':
    FILEPATHS = {
        "base": "./saved_models/cifar10_ResNet29v2_base.200.h5",
        "IHL": "./saved_models/cifar10_ResNet29v2_IHL.h5",
        "CHL": "./saved_models/cifar10_ResNet29v2_CHL.h5",
        "EKL": "./saved_models/cifar10_ResNet29v2_EKL.h5",
    }

    FILEPATHS_I = {
        "base": "./saved_models/ImageNet_ResNetv2_base.h5",
        "EKL": "./saved_models/ImageNet1000_ResNetv2_EKL.h5",
    }

    WEIGHTED_LOSS_FILE = "../confusion_matrices/cifar10_EKL_cmat.npy"
    CIFAR10C_DIR = "/data/alberto/datasets/CIFAR-10-C/"

    ####################### LOAD DATA ###########################
    #############################################################
    import torchvision.datasets as dset
    import torchvision.transforms as trn
    import torch

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    types = ["base", "EKL"]

    TRAIN_DATA_PATH = "/data/alberto/datasets/Tiny-ImageNet-C/"

    distortion_name = "brightness"

    for type in types:
        model = InceptionResNetV2()
        model = multi_gpu_model(model, 2)
        model.load_weights(FILEPATHS_I[type])

        for severity in range(1, 6):
            distorted_dataset = dset.ImageFolder(
                root=TRAIN_DATA_PATH + distortion_name + '/' + str(severity),
                transform=trn.Compose([trn.Resize(299), trn.ToTensor(), trn.Normalize(mean=mean, std=std)]))

            distorted_dataset_loader = torch.utils.data.DataLoader(
                distorted_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

            for x, y in distorted_dataset_loader:
                x = x.view(-1, 299, 299, 3).numpy()
                # x_prep = preprocess_input(x)
                predicted_x = model.predict(x)

                correct = np.argmax(predicted_x, 1) == y.numpy()
                accuracy = sum(correct) / len(correct)
                print(accuracy)

    exit()
    do_cifar10c()
