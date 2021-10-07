import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import ImageFilter
from losses import WeightedCC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

"""## The Grad-CAM algorithm """

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

    # The local path to our target image
    img_path = keras.utils.get_file(
        "african_elephant.jpg", " https://i.imgur.com/Bvro0YD.png"
    )

    # from ImagenetManualLoad import ImagenetManualLoad
    img_size = (299, 299)

    # inet = ImagenetManualLoad()
    # X_test, y_test = inet.get_X_Y_ImageNet("val", n=1000)

    model_builder = keras.applications.xception.Xception
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions
    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Make model -----------------------------------------------
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
    # -----------------------------------------------------------

    # Print what the top predicted class is
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # We load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

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
    # jet_heatmap = jet_heatmap.filter(ImageFilter.GaussianBlur(10))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "elephant_cam.jpg"
    superimposed_img.save(save_path)

    print("Saved in: "+save_path)
