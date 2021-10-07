import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import ImageFilter
from losses import WeightedCC
from tensorflow.keras.optimizers import Adam
from ImagenetManualLoad import ImagenetManualLoad
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

import os
import pickle
from pathlib import Path
from tqdm import tqdm

"""## The Grad-CAM algorithm """
# Code partially extracted from https://keras.io/examples/vision/grad_cam/

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def make_EKL_model(d):
    weight_matrix = np.load(d["EKL"]["WEIGHTED_LOSS_FILE"])
    wcc = WeightedCC(weights=weight_matrix)
    # wcc = WeightedCC(weights=weight_matrix, scaling=4)
    explicable_loss = wcc.get_loss()

    model = InceptionResNetV2()
    model = multi_gpu_model(model, 2)
    model.load_weights(d["EKL"]["EVAL_MODEL_FNAME"])
    model = model.layers[-2]
    model.compile(
        loss=explicable_loss,
        optimizer=Adam(),
        metrics=["accuracy"])

    return model

if __name__ == '__main__':

    last_conv_layer_name = "conv_7b_ac"
    classifier_layer_names = [
        "avg_pool",
        "predictions",
    ]

    d = {
        "EKL": {
            "WLF": True,
            "EVAL_MODEL_FNAME": "saved_models/ImageNet1000_ResNetv2_EKL.h5",
            "WEIGHTED_LOSS_FILE": "confusion_matrices/ImageNet_EKL_cmat.npy",
        },
    }

    img_size = (299, 299)
    N = 500

    inet = ImagenetManualLoad()
    preprocess_input = keras.applications.xception.preprocess_input

    X_test, y_test = inet.get_X_Y_ImageNet("val", preprocess=False, n=N)
    X_test_preproc = preprocess_input(X_test)

    # Make models and save image information ---------------------
    EKL_model = make_EKL_model(d)
    base_model = InceptionResNetV2()
    base_model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"])

    models = [base_model, EKL_model]
    EKL_cmat = np.load(d["EKL"]["WEIGHTED_LOSS_FILE"])

    aux_dict_f = "./tmp/aux_dict_{}.p".format(N)
    with open("./misc/imagenet_class_names", "r") as file:
        labels = [line.strip().split(",")[0] for line in file]
    labels_d = dict(zip(range(len(labels)), labels))

    if not os.path.exists(aux_dict_f):
        # Make predictions with both models
        EKL_preds = np.argmax(EKL_model.predict(X_test_preproc), 1)
        base_preds = np.argmax(base_model.predict(X_test_preproc), 1)

        aux_dict = {"both_incorrect": [[],[]], "EKL_correct": [[],[]], "both_correct": [[],[]]}

        for i, X in enumerate(tqdm(X_test)):
            true_pred = y_test[i]
            base_pred = base_preds[i]
            EKL_pred = EKL_preds[i]

            y_test_label = labels_d[y_test[i]]
            y_base_label = labels_d[base_preds[i]]
            y_EKL_label = labels_d[EKL_preds[i]]

            name = "{}_{}_{}.jpg".format(y_test_label, y_base_label, y_EKL_label)

            if EKL_pred != true_pred and base_pred != true_pred:
                if EKL_cmat[true_pred, base_pred] + 1.5 >= EKL_cmat[true_pred, EKL_pred]:
                    aux_dict["both_incorrect"][0].append(X)
                    aux_dict["both_incorrect"][1].append(name)
            elif EKL_pred == true_pred:
                if base_pred != true_pred:
                    aux_dict["EKL_correct"][0].append(X)
                    aux_dict["EKL_correct"][1].append(name)
                else:
                    aux_dict["both_correct"][0].append(X)
                    aux_dict["both_correct"][1].append(name)

        pickle.dump(aux_dict, open(aux_dict_f, "wb"))
    else:
        aux_dict = pickle.load(open(aux_dict_f, "rb"))
    # -----------------------------------------------------------

    # Create tmp folders
    [Path("./tmp/" + key).mkdir(parents=True, exist_ok=True) for key in aux_dict]

    i = -1
    for key in aux_dict.keys():
        X_test = np.array(aux_dict[key][0])
        y_names = aux_dict[key][1]

        for img, img_array, name in zip(X_test, preprocess_input(X_test), y_names):
            grid = []
            i += 1

            grid.append(img / 255)

            img_array = np.expand_dims(img_array, 0)
            for model in models:
                # Generate class activation heatmap
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)

                # We rescale heatmap to a range 0-255
                heatmap = np.uint8(255 * heatmap)

                # We use jet colormap to colorize heatmap
                jet = cm.get_cmap("jet")

                # We use RGB values of the colormap
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]

                # We create an image with RGB colorized heatmap
                jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
                jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

                # Superimpose the heatmap on original image
                superimposed_img = jet_heatmap * 0.8 + img
                superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
                superimposed_img = keras.preprocessing.image.img_to_array(superimposed_img)
                grid.append(superimposed_img / 255)

            # Save the superimposed image
            grid = np.vstack(grid)

            save_file = "./tmp/{}/{}_{}".format(key, i, name)
            plt.imsave(save_file, grid)
            print("[+]: Saved image in " + save_file)
