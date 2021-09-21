from __future__ import print_function
import os
import tensorflow.keras as keras
import argparse
import numpy as np
import resnet_models_comparison
import matplotlib.pyplot as plt
import seaborn as sns
import utils

from sklearn.preprocessing import normalize
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from cifarvgg.cifar10vgg import cifar10vgg
from cifarvgg.cifar100vgg import cifar100vgg
from ImagenetManualLoad import ImagenetManualLoad
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from imagenet_resnet_experiments import compile_WLF_model
from utils import get_ImageNet_EKL_cmat

def is_egregious_mistake(true, pred_v, preds, cmats, n_max=1, egreg=0):
    n_egreg = 0
    for pred_wlf, cmat in zip(preds, cmats):
        if cmat[true, pred_v] < cmat[true, pred_wlf]-egreg:
            n_egreg += 1

    return True if n_egreg >= n_max else False


def display_model_examples(model_v, model_wlf, dataset, cmat):
    NRUNS = 10
    for i_run in range(NRUNS):
        save_file = "./images/instance_comparison_{}_{}.png".format(dataset, i_run)

        with open("{}_class_names".format(dataset), "r") as file:
            labels = [line.strip() for line in file]

        labels_d = dict(zip(range(len(labels)), labels))

        y_pred_wlf = model_wlf.predict(x_test)
        y_pred_v = model_v.predict(x_test)

        display_x = []
        display_true = []
        display_pred_v = []
        display_pred_wlf = []
        n = len(y_test)
        for i in range(n):
            true = np.argmax(y_test[i])
            pred_v = np.argmax(y_pred_v[i])
            pred_wlf = np.argmax(y_pred_wlf[i])
            if pred_v != true and pred_wlf != true:
                if is_egregious_mistake(true, pred_v, pred_wlf, cmat):
                    display_x.append(x_test_untouched[i])
                    display_true.append(true)
                    display_pred_v.append(pred_v)
                    display_pred_wlf.append(pred_wlf)

        grid_size = 4
        fig, axes1 = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        j = 0
        i = 0
        c = 0
        while c < 16:

            if j == grid_size:
                i, j = i + 1, 0

            x_i = np.random.choice(range(len(display_x)))

            axes1[i, j].set_axis_off()
            axes1[i, j].imshow(display_x[x_i: x_i + 1][0])
            axes1[i, j].set_title(
                "T:{}\nL:{}:{:.4f} \nW/L:{}:{:.4f}".format(
                    labels_d[display_true[x_i]],
                    labels_d[display_pred_v[x_i]],
                    cmat[display_true[x_i], display_pred_v[x_i]],
                    labels_d[display_pred_wlf[x_i]],
                    cmat[display_true[x_i], display_pred_wlf[x_i]]
                )
            )
            c += 1
            j += 1

        fig.set_figheight(16)
        fig.set_figwidth(10)
        print("[+]: Saving {}".format(save_file))
        plt.savefig(save_file)


def display_model_examples_multiple(x_test, y_test, y_pred_v, pred_wlf, dataset, cmats, names):
    FIGHEIGHT = 22
    FIGWIDTH = 16
    FIGSIZE = (8, 8)
    egregiousness = 0.001 #0.2 if dataset == "imagenet" else 0
    grid_size = 4

    pred_file = "./human_studies/preds/preds_{}_true_base_wlf_egreg_{}.txt".format(dataset, egregiousness)
    pred_file = open(pred_file, "a")

    pred_wlf = [pred_wlf] if len(names) == 1 else pred_wlf
    n_egregious_max = len(names)

    print("[+]: Displaying egregious predictions")
    orig_dataset = "cifar100" if "cifar100" in dataset else "cifar10"
    with open("./misc/{}_class_names".format(orig_dataset), "r") as file:
        labels = [line.strip().split(",")[0] for line in file]
    labels_d = dict(zip(range(len(labels)), labels))

    display_x = []
    display_true = []
    display_pred_v = []
    display_pred_wlf = []

    for i, preds in enumerate(zip(*pred_wlf)):
        true = y_test[i]
        pred_v = y_pred_v[i]
        if np.all(pred_v != true) and np.all(np.array(preds) != true):
            if is_egregious_mistake(true, pred_v, preds, cmats, n_egregious_max, egregiousness):
                display_x.append(x_test[i])
                display_true.append(true)
                display_pred_v.append(pred_v)
                display_pred_wlf.append(preds)

                folder = "./human_studies/{}/egregious_examples/".format(dataset)
                if dataset == "imagenet":
                    image_name = "{}--{}_{}--{}_{}--{}_{}.jpg".format(i, true, labels_d[true],
                                                                      pred_v, labels_d[pred_v],
                                                                      preds[0], labels_d[preds[0]])

                elif "cifar10" in dataset or "cinic" in dataset:
                    image_name = "{}-T{}_{}-V{}_{}-I{}_{}-C{}_{}-E{}_{}.jpg".format(i, true, labels_d[true],
                                                                                    pred_v, labels_d[pred_v],
                                                                                    preds[0], labels_d[preds[0]],
                                                                                    preds[1], labels_d[preds[1]],
                                                                                    preds[2], labels_d[preds[2]])
                plt.imsave(folder+image_name, x_test[i])
                pred_file.write(image_name+"\n")

    times = len(display_x)//grid_size**2
    for k in range(times):
        save_file = "./human_studies/{}/egregious_examples/egregious_{}_{}.png".format(dataset, dataset, k)

        fig, axes1 = plt.subplots(grid_size, grid_size, figsize=FIGSIZE)
        i, j, c = 0, 0, 0
        while c < 16:
            if j == grid_size:
                i, j = i + 1, 0
            x_i = k*grid_size**2 + c

            title = "T: {}\n V: {}\n".format(labels_d[display_true[x_i]], labels_d[display_pred_v[x_i]])
            preds = list(display_pred_wlf[x_i])
            for n, p in zip(names, preds):
                title += "{}: {}\n".format(n, labels_d[p])

            axes1[i, j].set_axis_off()
            axes1[i, j].imshow((display_x[x_i: x_i + 1][0]))
            axes1[i, j].set_title(title)
            c += 1
            j += 1

        fig.set_figheight(FIGHEIGHT)
        fig.set_figwidth(FIGWIDTH)
        print("[+]: Saving {}".format(save_file))
        plt.savefig(save_file)


def get_stats(vanilla_cifarX_VGG, WLF_cifarX_VGG):
    y_pred_v = vanilla_cifarX_VGG.predict(x_test)
    y_pred_wlf = WLF_cifarX_VGG.predict(x_test)

    y_labels_wlf, y_labels_v, y_labels = np.argmax(y_pred_wlf, axis=1), \
                                         np.argmax(y_pred_v, axis=1), \
                                         np.argmax(y_test, axis=1)

    n = y_test.shape[0]
    counters = [0] * 6
    for y_true, pred_y, pred_wlf_y in zip(y_labels, y_labels_v, y_labels_wlf):
        if pred_y == y_true and pred_wlf_y == y_true:
            counters[0] += 1
        if pred_y != y_true and pred_wlf_y == y_true:
            counters[1] += 1
        if pred_y != y_true and pred_wlf_y != y_true and cmat[y_true, pred_wlf_y] > cmat[y_true, pred_y]:
            counters[2] += 1
        if pred_y == y_true and pred_wlf_y != y_true:
            counters[3] += 1
        if pred_y != y_true and pred_wlf_y != y_true and cmat[y_true, pred_wlf_y] < cmat[y_true, pred_y]:
            counters[4] += 1
        if pred_y != y_true and pred_wlf_y != y_true and cmat[y_true, pred_wlf_y] == cmat[y_true, pred_y]:
            counters[5] += 1

    counters = np.array(counters) / n

    # Correct Correct | Incorrect Correct | Inc
    print("WLF model | Vanilla")
    print("""Correct-Correct: {}\n 
          Incorrect-Correct: {}\n
          Incorrect-Incorrect(closer): {}\n
          Correct-Incorrect: {}\n
          Incorrect(closer)-Incorrect: {}\n
          Incorrect-Incorrect: {}\n""".format(*counters))


def plot_grid_egregious_images_imagenet():
    n = 50000
    names = ["EKL"]
    dataset_name = "imagenet"
    cmats = [get_ImageNet_EKL_cmat()]
    pv_f, pekl_f = ["./tmp/Imagenet_pred_v.npy", "./tmp/Imagenet_pred_ekl.npy"]

    config = ImagenetManualLoad().get_config()
    NGPUs = int(config["CONST"]["n_gpu"])

    obj = ImagenetManualLoad()
    X, Y = obj.get_X_Y_ImageNet("val", preprocess=True, n=n)
    X_untouched = obj.get_X("val", preprocess=False, n=n)
    Y = to_categorical(Y, 1000)

    if not os.path.exists(pekl_f):
        model_v = InceptionResNetV2()
        model_v = compile_WLF_model(model_v, "EKL")

        model_ekl = multi_gpu_model(InceptionResNetV2(), NGPUs) if NGPUs > 1 else InceptionResNetV2()
        model_ekl.load_weights("./saved_models/ImageNet1000_ResNetv2_EKL.h5")
        model_ekl = compile_WLF_model(model_ekl, "EKL")

        # Predict
        pred_v, pred_ekl = model_v.predict(X), model_ekl.predict(X)

        np.save(pv_f, pred_v)
        np.save(pekl_f, pred_ekl)

    else:
        pred_v, pred_ekl, = np.load(pv_f), np.load(pekl_f)

    pred_v = np.argmax(pred_v, 1)
    pred_wlf = np.argmax(pred_ekl, 1)
    pred_wlf = list(pred_wlf)

    Y = np.argmax(Y, 1)
    display_model_examples_multiple(X_untouched, Y, pred_v, pred_wlf, dataset_name, cmats, names)


def plot_grid_egregious_images_cifarX():
    dataset_name = args.dataset
    num_classes = 100 if "cifar100" in dataset_name else 10

    ##############################################################
    ##############################################################
    # Load the CIFAR data.
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train_mean = np.mean(x_train.astype("float32")/255., axis=0)

    x_test, y_test = utils.load_cifarX(dataset_name)
    x_test_untouched = x_test.copy()
    x_test -= x_train_mean

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Convert class vectors to binary class matrices.
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    ####################### PREPARE MODELS #######################
    ##############################################################

    model_to_use = "VGG"
    if model_to_use == "VGG":
        cifarX_dataset = cifar100 if "cifar100" in dataset_name else cifar10
        cmat_fname = "cifar10_EKL_cmat.npy" if dataset_name == "cifar10" else "cifar100_EKL_cmat.npy"
        cmat_fname = os.path.join(CMATS_FOLDER, cmat_fname)

        VGG_trained_model = None
        if not VGG_trained_model:
            VGG_trained_model = "cifar10_cifarvgg10_wlf_wordnet_200.h5" if dataset_name == "cifar10" \
                else "cifar100_cifarvgg100_wlf_wordnet.h5"

        cifarX_VGG = cifar10vgg if dataset_name == "cifar10" else cifar100vgg
        # Get vanilla CifarX VGG
        vanilla_cifarX_VGG = cifarX_VGG(train=False)

        # Get WLF-C10VGG
        WLF_cifarX_VGG = cifarX_VGG(train=False, weights_file='saved_models/{}'.format(VGG_trained_model)).model

        ##############################################################
        ##############################################################

        # Adversarially perturbed
        x_test = np.load("tmp/c100_FGSM_x_test.npy")

        cmat = np.load(cmat_fname)
        display_model_examples(vanilla_cifarX_VGG, WLF_cifarX_VGG, dataset_name, cmat)
        get_stats(vanilla_cifarX_VGG, WLF_cifarX_VGG)

        ##############################################################
        ##############################################################

    elif model_to_use == "ResNet":
        names = ["IHL", "CHL", "EKL"]
        # Preprocess data if ResNetv2 is selected
        # Normalize data.
        # x_train = x_train.astype("float32") / 255
        # x_test = x_test.astype("float32") / 255

        # If subtract pixel mean is enabled
        # x_train_mean = np.mean(x_train, axis=0)
        # x_train -= x_train_mean
        # x_test -= x_train_mean

        depth = 3 * 9 + 2
        model_v = resnet_models_comparison.resnet_v2(input_shape=input_shape, depth=depth)
        filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), "cifar10_ResNet29v2_base.200.h5")
        model_v.load_weights(filepath)

        models, cmats = [], []
        for name in names:
            cmat = np.load(os.path.join(CMATS_FOLDER, "cifar10_{}_cmat_normalized.npy".format(name)))
            filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), "cifar10_ResNet29v2_{}.h5".format(name))
            model = resnet_models_comparison.resnet_v2(input_shape=input_shape, depth=depth)
            model.load_weights(filepath)

            models.append(model)
            cmats.append(cmat)

        pred_wlf = [np.argmax(model.predict(x_test), 1) for model in models]
        pred_v = np.argmax(model_v.predict(x_test), 1)
        display_model_examples_multiple(x_test_untouched, y_test, pred_v, pred_wlf, dataset_name, cmats, names)


if __name__ == '__main__':
    CMATS_FOLDER = "confusion_matrices/"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--VGG_trained_model', type=str, default=False,
                        help='CifarX saved model name to compare to the original')
    parser.add_argument('--dataset', type=str, default="cifar100", help='Name of the dataset')
    parser.add_argument('--config', type=str, help='Config file')

    args = parser.parse_args()

    dataset = args.dataset
    if "cifar" in dataset or "cinic" in dataset:
        plot_grid_egregious_images_cifarX()

    elif "imagenet" in dataset:
        plot_grid_egregious_images_imagenet()

##############################################################
##############################################################
# from metric_utils import plot_confusion_matrix
# plot_confusion_matrix(cmat, dataset_name)
# second_important = cmat.argsort()[..., ::-1][:, 1]

# same_miscl = []
# same_miscl.append((labels_d[true], labels_d[pred_v], labels_d[preds[0]]))

