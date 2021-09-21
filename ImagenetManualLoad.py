import os
import glob
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from configparser import ConfigParser, ExtendedInterpolation
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from matplotlib.pyplot import imsave

INPUT_SIZE = (299, 299)  # Default input size for ResNetv2-ImageNet

class ImagenetManualLoad():
    def __init__(self):
        self.config = self.get_config()
        self.dataset = self.config["CONST"]["dataset"]

    def save_list_of_imgs_to_folder(self, path, type, l):
        dir = os.path.join(path, self.config["DIRS"]["images"])
        img_file = "{}.jpg"
        savefile = os.path.join(dir, img_file)

        n = int(max(l))+1
        X = self.get_X(type, preprocess=False, n=n)

        [imsave(savefile.format(ii), X[int(i)]) for ii, i in enumerate(l)]

    def gen_sample_grid_imgs_from_label(self, path, label, grid_size=6):
        import torch
        from torchvision.utils import make_grid

        dir = os.path.join(path, self.config["DIRS"]["images"])
        img_file = "{}.jpg".format(label)
        save_file = os.path.join(dir, img_file)

        if os.path.exists(save_file):
            print("[+]: Grid {} was already saved".format(save_file))
            return

        try:
            os.makedirs(dir)
        except:
            pass

        X, _ = self.get_specific_class_X_Y(label, grid_size*grid_size)
        X = X.transpose((0,3,1,2))
        X = make_grid(torch.tensor(X), grid_size).numpy().transpose((1,2,0))
        imsave(save_file, X)
        print("[+]: Saved grid image {}".format(save_file))

    def get_specific_class_X_Y(self, label, n):
        syn_dict = self.load_dict()
        filelist = glob.glob('{}/{}/*.JPEG'.format(self.config[self.dataset]["train_generator"], label))

        if not len(filelist):
            print("[-]: Images for class label {} not found. Exiting ...".format(label))
            exit()

        x_aux, y_aux = [], []
        for i, fname in enumerate(tqdm(filelist, total=n-1)):
            img = np.array(Image.open(fname).resize((INPUT_SIZE)))
            if len(img.shape) == 2:
                img = np.tile(img[..., None], [1, 1, 3])

            x_aux.append(img)
            y_aux.append(syn_dict[label]["label"])

            if n > 0 and i == n - 1:
                break
        X, Y = np.array(x_aux), np.array(y_aux)
        return X, Y

    def get_config(self):
        parser = argparse.ArgumentParser(description="ImagenetManualLoad, please input config file")
        parser.add_argument('--config', type=str)
        args = parser.parse_args()

        config = ConfigParser(interpolation=ExtendedInterpolation())
        if args.config:
            config.read(args.config)
        else:
            config.read("configs/base.ini")
        return config

    def get_val_gt_labels_from_csv(self, csv_file):
        syn_dict = self.load_dict()
        filename = self.config["DIRS"]["tmp"] + "y_gt_val_imagenet.npy"
        if not os.path.exists(filename):
            df = pd.read_csv(csv_file)
            df = df.iloc[:, 1]

            gt_labels = []
            for i in tqdm(range(df.shape[0])):
                synset = df.iloc[i].split(" ")[0]
                label = syn_dict[synset]["label"]
                gt_labels.append(label)

            ret = np.vstack(gt_labels).squeeze()
            np.save(filename, ret)
            print("[+]: {} saved".format(filename))
            return ret

        print("[+]: Loading {}".format(filename))
        return np.load(filename)

    def get_X(self, type, preprocess=True, save_new=False, n=-1):
        preproc_name = "_preproc" if preprocess else ""
        quant = "_{}".format(n) if n > 0 else ""
        x_filename = self.config["DIRS"]["tmp"] + "x_{}_imagenet{}{}.npy".format(type, preproc_name, quant)

        if not os.path.exists(x_filename) or save_new:
            filelist = sorted(glob.glob('{}*.*'.format(self.config["IMAGENET"]["val"])))
            aux = []
            for i, fname in enumerate(tqdm(filelist)):
                img = np.array(Image.open(fname).resize((INPUT_SIZE)))
                if len(img.shape) == 2:
                    img = np.tile(img[..., None], [1, 1, 3])
                aux.append(img)

                if n > 0 and i == n - 1:
                    break

            X = np.array(aux)
            X = preprocess_input(X) if preprocess else X

            np.save(x_filename, X)
            print("[+]: {} saved".format(x_filename))

        else:
            print("[+]: Loading {}".format(x_filename))
            X = np.load(x_filename)

        return X

    def get_X_Y_ImageNet(self, type, preprocess=True, save_new=False, n=-1):
        DATASET = self.config["CONST"]["dataset"]

        preproc_name = "_preproc" if preprocess else ""
        quant = "_{}".format(n) if n > 0 else ""
        x_filename = self.config["DIRS"]["tmp"] + "x_{}_imagenet{}{}.npy".format(type, preproc_name, quant)
        y_filename = self.config["DIRS"]["tmp"] + "y_{}_imagenet.npy".format(type)

        if type == "val":
            X = self.get_X(type, preprocess, save_new, n)
            n = X.shape[0] if n == -1 else n
            Y = self.get_val_gt_labels_from_csv(self.config["IMAGENET"]["solution_file"])[:n]
            return X, Y

        elif type == "train":
            if not os.path.exists(x_filename) or save_new:
                dataset = self.config["IMAGENET"]["train_2012"] if "2012" in type else self.config[DATASET]["train_all"]
                syn_dict = self.load_dict()
                filelist = glob.glob('{}*.jpg'.format(dataset))
                x_aux, y_aux = [], []
                for i, fname in enumerate(tqdm(filelist)):
                    img = np.array(Image.open(fname).resize((INPUT_SIZE)))
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], [1, 1, 3])
                    fname = os.path.basename(fname)
                    syn = fname.split("_")[0]

                    x_aux.append(img)
                    y_aux.append(syn_dict[syn]["label"])

                    if n > 0 and i == n - 1:
                        break

                X, Y = np.array(x_aux), np.array(y_aux)
                if preprocess:
                    X = preprocess_input(X)

                np.save(x_filename, X)
                print("[+]: {} saved".format(x_filename))
                np.save(y_filename, Y)
                print("[+]: {} saved".format(y_filename))
                return X, Y

            print("[+]: Loading {} and {}".format(x_filename, y_filename))
            return np.load(x_filename), np.load(y_filename)

    def load_dict(self):
        return_dict = {}
        with open(self.config["IMAGENET"]["synset_mapping_file"], 'r') as inf:
            for i, line in enumerate(inf):
                syn_id = line.split(" ")[0]
                name = line.split(syn_id + " ")[1].strip()
                return_dict[syn_id] = {"label": i, "synset": name}
        return return_dict