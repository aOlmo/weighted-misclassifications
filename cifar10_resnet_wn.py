"""
This bulk of this code originates from Keras' Resnet for CIFAR 10: https://keras.io/examples/cifar10_resnet/
We modify the loss function and the evaluation metric depending on the classifier to be trained.
"""

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

from resnetv2 import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from losses import WeightedCC

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


""" Setting Hyperparameters """
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |   | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     | n | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080
#           |   | %Accuracy | %Accuracy | %Accuracy | %Accuracy | v2
# ----------------------------------------------------------------------------
# ResNet20  | 3 | 92.16     | 91.25     | 91.85     | -----     | 120
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2

# Computed depth from supplied model parameter n
depth = n * 9 + 2

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

# Convert class vectors to binary class matrices.
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


model = resnet_v2(input_shape=input_shape, depth=depth)

# Read configuration for this run
MODEL_FILE = "cifar10_ResNet29v2_EKL.h5"
WEIGHTED_LOSS_FILE = "confusion_matrices/cifar10_EKL_cmat.npy"
IMAGE_FILE = "images/cmat_c10_ResNetv2_WLF_trained.png"

# Decide which loss function to use.
if WEIGHTED_LOSS_FILE:
    weight_matrix = np.load(WEIGHTED_LOSS_FILE)
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

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = MODEL_FILE
filepath = os.path.join(save_dir, model_name)

# try:
#     model.load_weights(filepath)
#     print("Loaded trained model at %s " % filepath)
#
# except:
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.

print("Not using data augmentation.")
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=callbacks,
)

# model.save(filepath)
print("Saved trained model at %s " % filepath)

# Accuracy and Confusion-matrix based evaluations
# metric_utils.evaluate(
#     model, x_test, y_test, image_file_name=IMAGE_FILE,
# )
