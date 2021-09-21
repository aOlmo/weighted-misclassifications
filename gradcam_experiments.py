import os
import cv2
import keras
import numpy as np
import tensorflow as tf

from losses import WeightedCC
from keras.optimizers import Adam
from keras.datasets import cifar10

from tqdm import tqdm
from keras import backend as K
from matplotlib import pyplot as plt
from keras.preprocessing import image
from resnet_models_comparison import *
from tensorflow.python.framework import ops

# Partial code used from https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb

c10 = ["airplane", "automobile", "bird", "cat", "deer",
       "dog", "frog", "horse", "ship", "truck"]
savedir = "gradcam_results/"

def compile_model(model, weighted_loss_file):
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

    return model


def get_cifar10_ready_data():
    num_classes = 10

    # Cifar-10 loading of data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def build_model(weights_file):
    n = 3
    depth = n * 9 + 2
    input_shape = x_train.shape[1:]
    filepath = os.path.join(os.path.join(os.getcwd(), "saved_models"), weights_file)

    model = resnet_v2(input_shape=input_shape, depth=depth)
    model.load_weights(filepath)

    return model


def preprocess_input(x):
    x = x.astype("float32") / 255
    return x
    # return x - np.mean(x, axis=0)


#############################################################################
# --------------------------- Utility functions --------------------------- #
#############################################################################

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # if K.image_dim_ordering() == 'th':
    #     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


###########################################################################
# --------------------------- Guided Backprop --------------------------- #
###########################################################################
def build_guided_model(weights_file):
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model(weights_file)
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


###################################################################
# --------------------------- GradCAM --------------------------- #
###################################################################
def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (W, H), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def compute_saliency(model, guided_model, layer_name, img_path, savefile, w_gb=False, save=False):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    raw_img = plt.imread(img_path)
    preprocessed_input = load_image(img_path)

    cls = np.argmax(model.predict(preprocessed_input))
    print("Predicted: {}".format(c10[cls]))

    savefile += "_{}".format(c10[cls])

    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2

    gradcam = np.uint8(jetcam)
    gb = deprocess_image(gb[0])
    gg = deprocess_image(guided_gradcam[0])

    if w_gb:
        # Add separation lines to print
        # z0 = np.zeros((H, W + 1, 3))
        # z0[:, :-1] = raw_img

        # Add separation lines to print
        z1 = np.zeros((H, W + 1, 3))
        z1[:, :-1] = gradcam

        z2 = np.zeros((H, W + 1, 3))
        z2[:, :-1] = gb

        img = np.hstack((z1, z2, gg))

    else:
        z1 = np.zeros((H, W + 1, 3))
        z1[:, :-1] = gradcam
        img = z1

    if save:
        plt.imsave(os.path.join(savedir, "grid_test.jpg"), img / 255)
        savename = "{}.jpg".format(os.path.join(savedir, savefile))
        cv2.imwrite(savename, img)
        print("[+]: Saved img in ", savename)

    visualize = False
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return img


###################################################################
# ----------------------------- MAIN ---------------------------- #
###################################################################

if __name__ == '__main__':

    aux_file = os.path.join(savedir, 'current_img.jpg')
    names = ["base", "EKL", "CHL"]

    model_files = [
        "cifar10_ResNet29v2_base.200.h5",
        "cifar10_ResNet29v2_EKL.h5",
        "cifar10_ResNet29v2_CHL.h5"]

    cmat_folder = "confusion_matrices/"

    H, W = 32, 32  # Input shape, defined by the model (model.input_shape)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    EKL_cmat = np.load(cmat_folder + "cifar10_EKL_cmat.npy")
    CHL_cmat = np.load(cmat_folder + "cifar10_CHL_cmat.npy")
    IHL_cmat = np.load(cmat_folder + "cifar10_IHL_cmat.npy")

    models, guided_models = [], []
    for i, (name, mf) in enumerate(zip(names, model_files)):
        models.append(build_model(mf))
        K.reset_uids()
        guided_models.append(build_guided_model(mf))
        K.reset_uids()

    w_gb = False
    layer_name = "conv2d_31"
    selected_dict = {
        "get_WLF_close_and_base_incorrect": [701, 67, 1521, 2414],
        "get_WLF_correct_and_base_incorrect": [3, 62, 74, 142],
        "get_WLF_correct_and_base_correct": [2, 18, 13]
    }

    for key in selected_dict.keys():
        img_grid_flag = key
        # for i in tqdm(range(x_test.shape[0])):
        for i in tqdm(selected_dict[key]):
            grid = []
            suffix = ""
            true_pred = int(y_test[i])
            plt.imsave(aux_file, x_test[i] / 255)

            base_pred = int(np.argmax(models[0].predict(load_image(aux_file))))
            EKL_pred = int(np.argmax(models[1].predict(load_image(aux_file))))
            CHL_pred = int(np.argmax(models[2].predict(load_image(aux_file))))

            if img_grid_flag == "get_WLF_correct_and_base_incorrect":
                if CHL_cmat[true_pred, base_pred] + 1 >= CHL_cmat[true_pred, EKL_pred] or \
                        CHL_cmat[true_pred, base_pred] + 1 >= CHL_cmat[true_pred, CHL_pred]:
                    continue
                suffix = "_we_correct"
            elif img_grid_flag == "get_WLF_close_and_base_incorrect":
                if (CHL_cmat[true_pred, base_pred] + 1 >= CHL_cmat[true_pred, EKL_pred] or
                    CHL_cmat[true_pred, base_pred] + 1 >= CHL_cmat[true_pred, CHL_pred]) or \
                        (CHL_pred == true_pred or EKL_pred == true_pred):
                    continue
                suffix = "_we_closer"
            elif img_grid_flag == "get_WLF_correct_and_base_correct":
                if true_pred != base_pred or true_pred != EKL_pred or true_pred != CHL_pred:
                    continue
                suffix = "_all_correct"

            print("\n[+]: True class ", c10[true_pred])

            raw_img = plt.imread(aux_file)
            z0 = np.zeros((H, W + 1, 3))
            z0[:, :-1] = raw_img
            for j, (name, model, guided_model) in enumerate(zip(names, models, guided_models)):
                savefile = "{}_{}".format(name, i)
                img = compute_saliency(model, guided_model, layer_name, aux_file, savefile, w_gb=False, save=False)
                grid.append(img)

            if len(grid) == 3:
                if w_gb:
                    grid = np.vstack(grid)
                else:
                    grid = np.vstack((z0, np.vstack(grid)))
                print("[+]: Saving new grid")
                plt.imsave(os.path.join(savedir, "grid_img_{}_{}{}{}{}{}.jpg".format(
                    i, true_pred, base_pred, EKL_pred, CHL_pred, suffix)), grid / 255)
