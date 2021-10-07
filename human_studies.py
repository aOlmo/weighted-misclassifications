import os
import pdb
import sys
import numpy as np
import pandas as pd
import shutil
import random

from PIL import Image
from statistics import mean
from tensorflow.keras.datasets import cifar10
from matplotlib.pyplot import imsave
from Imagenet import Imagenet
from pdb import set_trace as st

random.seed(0)

N = 50
losses = ["base", "EKL", "CHL", "IHL"]
y_pred_file = "./saved_models/y_te_pred_c10_{}.npy"
classnames_file = "{}/pred_true.txt"
c10_classnames = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]

try:
    y_test = np.load("y_test.npy")
    x_test = np.load("x_test.npy")
except:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.squeeze()
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)


#################################### IMAGENET STUDIES ####################################
##########################################################################################

def save_grid_and_images_for_userstudy_imagenet():
    f = open("./human_studies/preds/preds_imagenet_true_base_ekl.txt", "r")

    obj = Imagenet()

    imgs_to_save = []
    for line in f:
        if line == '': continue
        ii, _, base, ekl = line.strip().split("--")
        imgs_to_save.append(int(ii))

        obj.gen_sample_grid_imgs_from_label("human_studies/imagenet/", ekl.split(" ")[0].split("_")[1], grid_size=3)
        obj.gen_sample_grid_imgs_from_label("human_studies/imagenet/", base.split(" ")[0].split("_")[1], grid_size=3)

    # obj.save_list_of_imgs_to_folder("human_studies/imagenet/", "val", imgs_to_save)


def save_within_subject_mturk_html_page_imagenet(url):
    """
    Within-Subjects design of the MTurk studies: Each subject is asked to give their opinion on each classifier
    :param url: Base URL where the images are going to be placed. Need to be accessed via Internet
    :return: Writes a new HTML suitable for the MTUrk studies
    """

    header = """<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />\
    <section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">\
    <div class="row col-xs-12 col-md-12"><!-- Instructions -->\
    <div class="panel panel-primary">\
    <div class="panel-heading"><strong>Instructions</strong></div>\
    <div class="panel-body">\
    <p>As a part of this HIT, you will be asked to answer questions about 52 images. For each image, select the most appropriate option.</p>\
    <p>This study is part of a research effort. We request you to answer the questions sincerely.</p>\
    <p><span data-darkreader-inline-color="" style="color: rgb(255, 0, 0); --darkreader-inline-color:#ff1a1a;"><strong>IMPORTANT ** YOU MUST ANSWER ALL QUESTIONS&nbsp;TO GET YOUR HIT APPROVED (TO RECEIVE PAYMENT). **</strong></span></p>\
    </div>\
    </div>\
    <!-- End Instructions --><!-- Content Body -->\
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <!--<button id="show">I agree and understand I have to answer all questions sincerely to get paid</button>
    
    <script>
        function showMenu() {
         $("#questions").show();   
         $("#show").hide();  
        }
        
        $('#show').click(showMenu);
    </script>  
    
    
    <section id='questions' style="display:none">\n\n -->
    <section>\
    """

    item_holder = """
        <div>\
            <img height="299" width="299" name="img_{0}_true_{3}" src="{1}/{0}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/>\
            <h4><b>Q{5}: Based on this image, how reasonable are the following misclassifications?</h4></b>\
            <table>\
                <tr>\
                    {2}\
                </tr>\
            </table><hr>\
        </div>"""

    column = """
            <td style='border: 1px solid aliceblue; background: aliceblue'>\
                <fieldset style='background: inherit'><label><span style="font-weight:normal"> Classifier\'s prediction: </span> {3} </label>\
                    <div class="radio"><label><input name="{0}_true_{4}_pred_{2}_{5}" type="radio" value="4" />Highly reasonable (understandable)</div></label>\
                    <div class="radio"><label><input name="{0}_true_{4}_pred_{2}_{5}" type="radio" value="3" />Somewhat reasonable</div></label>\
                    <div class="radio"><label><input name="{0}_true_{4}_pred_{2}_{5}" type="radio" value="2" />Not sure</div></label>\
                    <div class="radio"><label><input name="{0}_true_{4}_pred_{2}_{5}" type="radio" value="1" />Somewhat unreasonable</div></label>\
                    <div class="radio"><label><input name="{0}_true_{4}_pred_{2}_{5}" type="radio" value="0" />Highly unreasonable (surprised)</div></label>\
                </fieldset>\n\n\
                <b>Examples of {3} below</b>
                <img height="299" width="299" src="{1}/{2}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/>\
            </td>"""

    bait = """<div><img height="299" width="299" src="{1}/{0}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/></div>\
    <fieldset style="border: 1px solid black"><label>This is a: </label>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="circle" />Circle</div>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="triangle" />Triangle</div>\
    </fieldset>\n\n
    <hr>
    """

    footer = '\
             </section>\
    <!-- End Content Body --></div>\
    </section>\
    <style type="text/css">fieldset {padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }\
    </style>\n'

    # Generate page
    page = header

    f = open("./human_studies/preds/preds_imagenet_true_base_ekl.txt", "r")

    image_num = 0
    for line in f:
        image_num += 1
        data = ""
        image_name = line.split(".")[0]
        x_i, true, base, ekl = image_name.split("--")
        # Data for the base classifier string example: "700_n03887697 paper towel"
        true_label, true_name = true.split("_")[1].split(" ", 1)
        base_label, base_name = base.split("_")[1].split(" ", 1)
        ekl_label, ekl_name = ekl.split("_")[1].split(" ", 1)

        # st()

        data += column.format(image_num, url, base_label, base_name, true_label, "base")
        data += "<td style='white-space: nowrap;' width='200px'></td>"
        data += column.format(image_num, url, ekl_label, ekl_name, true_label, "ekl")
        if image_num == 12:
            page += bait.format('circle', url, image_num)
        elif image_num == 42:
            page += bait.format('triangle', url, image_num)

        page += item_holder.format(image_name, url, data, true_label, true_name, image_num)

    page += footer
    savepath = os.path.join("human_studies/imagenet/", "imagenet_study.html")
    print("[+]: Saving HTML in {}".format(savepath))
    html_page = open(savepath, "w")
    html_page.write(page)


def process_imagenet_mturk_userstudy(filename):
    prefix = ""
    filename = prefix + filename
    df = pd.read_csv(filename)

    n_rows = len(df)
    # Get the data columns
    df = df[df.columns[27:-2]]
    df = df[df["Answer.similarity_triangle_42"] == "triangle"]
    df = df[df["Answer.similarity_circle_12"] == "circle"]
    rem_rows = len(df)
    df = df[df.columns[:-2]]

    preds_dict = {"base": {}, "ekl": {}}
    for col in df:
        id, _, true_class, _, pred_class, base_ekl = col.split("_")
        id = int(id.split(".")[1])

        if not id in preds_dict[base_ekl]:
            preds_dict[base_ekl][id] = 0

        likert_score = df[col].mean()
        preds_dict[base_ekl][id] = likert_score

    print("Workers used {}/{} \n--".format(rem_rows, n_rows))

    print("Base: {}".format(mean(preds_dict["base"].values())))
    print("EKL: {}".format(mean(preds_dict["ekl"].values())))


#################################### CIFAR10 STUDIES #####################################
##########################################################################################
def get_same_miscl_imgs_cifar10():
    prefix = os.getcwd()
    y_miscl_prev = False
    for i, loss in enumerate(losses):
        y_pred = np.argmax(np.load(y_pred_file.format(loss)), 1)
        y_corr = y_pred - y_test
        y_miscl = np.nonzero(y_corr)[0]

        if i == 0:
            y_miscl_prev = y_miscl
            continue

        same_miscl = list(set(y_miscl).intersection(y_miscl_prev))
        y_miscl_prev = same_miscl

    selected = np.random.choice(same_miscl, N)
    save_selected_images_x_test_y_test(selected, prefix)


def get_preds_and_true_labels(path):
    file = open(path)
    true_labels, pred_labels = [], []
    for line in file:
        pred, true = line.strip().split(",")
        pred_labels.append(pred)
        true_labels.append(true)

    return pred_labels, true_labels


def save_selected_images_x_test_y_test(selected, prefix):
    for loss in losses:
        y_pred = np.argmax(np.load(y_pred_file.format(loss)), 1)
        pathdir = os.path.join(prefix, loss)

        try:
            os.mkdir(pathdir)
        except:
            pass

        print("[+]: Saving images for {}".format(loss))
        for i, img in enumerate(x_test[selected]):
            img = Image.fromarray(img)
            img.save("{}/{}.jpg".format(pathdir, i))

        print("[+]: Saving labels for {}".format(loss))
        file = open(classnames_file.format(pathdir), "w")
        for y_pr, y_te in zip(y_pred[selected], y_test[selected]):
            line = "{},{}\n".format(c10_classnames[y_pr], c10_classnames[y_te])
            file.write(line)


def save_between_subject_mturk_html_page_cifar10(url):
    """
    Between-subject design of the MTurk experiments: Here each subject gets to evaluate just one model.
    The images and their predictions have to be put in separate folders for each class.
    :param url: URL to input to the HTML file so that we can show the images
    :return: Writes an HTML page that can be used in MTurk
    """
    header = '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />\
    <section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">\
    <div class="row col-xs-12 col-md-12"><!-- Instructions -->\
    <div class="panel panel-primary">\
    <div class="panel-heading"><strong>Instructions</strong></div>\
    <div class="panel-body">\
    <p>As a part of this HIT, you will be asked to answer questions about 52 images. For each image, select the most appropriate option.</p>\
    <p>This study is part of a research effort. We request you to answer the questions sincerely.</p>\
    <p>** YOU MUST ANSWER ALL QUESTIONS TO GET YOUR HIT APPROVED (TO RECEIVE PAYMENT). **</p>\
    </div>\
    </div>\
    <!-- End Instructions --><!-- Content Body -->\
    <section>\n\n'
    item_holder = '<div><img height="42" width="42" src="{1}/{0}.jpg"/></div>\
    <fieldset><label>Q{0}: <span style="font-weight:normal"> Classifier\'s prediction: </span> {2} </label>\
    <div class="radio"><label><input name="likert_{0}" type="radio" value="0" />Highly unreasonable (surprised)</div></label>\
    <div class="radio"><label><input name="likert_{0}" type="radio" value="1" />Somewhat unreasonable</div></label>\
    <div class="radio"><label><input name="likert_{0}" type="radio" value="2" />Not sure</div></label>\
    <div class="radio"><label><input name="likert_{0}" type="radio" value="3" />Somewhat reasonable</div></label>\
    <div class="radio"><label><input name="likert_{0}" type="radio" value="4" />Highly reasonable (understandable)</div></label>\
    </fieldset>\n\n'
    bait = '<div><img height="42" width="42" src="{1}/{0}.jpg"/></div>\
    <fieldset><label>This is a: </label>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="circle" />Circle</div>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="triangle" />Triangle</div>\
    </fieldset>\n\n'

    footer = '</section>\
    <!-- End Content Body --></div>\
    </section>\
    <style type="text/css">fieldset {padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }\
    </style>\n'

    for name in losses:
        url = url.format(name)
        # Generate page
        page = header
        preds_file = open("human_studies/{}/pred_true.txt".format(name))
        for image_num in range(0, 50):
            if image_num == 12:
                page += bait.format('circle', url, image_num)
            elif image_num == 42:
                page += bait.format('triangle', url, image_num)

            y_pred = preds_file.readline().split(",")[0]
            page += item_holder.format(image_num, url, y_pred)

        page += footer
        html_page = open("human_studies/{}.html".format(name), "w")
        html_page.write(page)


def get_all_preds_and_true(path, losses):
    """
    Parses the predicted and true labels in the specified pathfile.
    :param path: Path to parse the data from
    :param losses: String to complete the pathfile
    :return: Both the true labels and predicted labels written in the specified txt file (path)
    """
    all_preds = []
    all_true = []

    for loss in losses:
        file = open(os.path.join(os.getcwd(), path.format(loss)))
        cur_preds = []
        cur_true = []
        for line in file:
            pred, true = line.split(",")
            cur_preds.append(pred)
            cur_true.append(true)
        all_preds.append(cur_preds)
        all_true.append(cur_true)

    return all_preds, all_true


def save_within_subject_mturk_html_page_cifar10(url):
    """
    Within-Subjects design of the MTurk studies: Each subject is asked to give their opinion on each classifier
    :param url: Base URL where the images are going to be placed. Need to be accessed via Internet
    :return: Writes a new HTML suitable for the MTUrk studies
    """

    header = '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />\
    <section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">\
    <div class="row col-xs-12 col-md-12"><!-- Instructions -->\
    <div class="panel panel-primary">\
    <div class="panel-heading"><strong>Instructions</strong></div>\
    <div class="panel-body">\
    <p>As a part of this HIT, you will be asked to answer questions about 52 images. For each image, select the most appropriate option.</p>\
    <p>This study is part of a research effort. We request you to answer the questions sincerely.</p>\
    <p>** YOU MUST ANSWER ALL QUESTIONS TO GET YOUR HIT APPROVED (TO RECEIVE PAYMENT). **</p>\
    </div>\
    </div>\
    <!-- End Instructions --><!-- Content Body -->\
    <section>\n\n'

    item_holder = '<div><img height="42" width="42" src="{1}/{0}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/>\
        <h5><b>Q{0}:</b> The above image represents a <b>{3}</b>. How reasonable are the following misclassifications?</h5>\
        <table>\
            <tr>\
                {2}\
            </tr>\
        </table>\
        </div>'

    column = '<td>\
                <fieldset><label><span style="font-weight:normal"> Classifier\'s prediction: </span> {1} </label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="0" />Highly unreasonable (surprised)</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="1" />Somewhat unreasonable</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="2" />Not sure</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="3" />Somewhat reasonable</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="4" />Highly reasonable (understandable)</div></label>\
                </fieldset>\n\n\
            </td>'

    bait = '<div><img height="60" width="60" src="{1}/{0}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/></div>\
    <fieldset><label>This is a: </label>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="circle" />Circle</div>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="triangle" />Triangle</div>\
    </fieldset>\n\n'

    footer = '</section>\
    <!-- End Content Body --></div>\
    </section>\
    <style type="text/css">fieldset {padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }\
    </style>\n'

    # Generate page
    page = header

    # TODO: Careful with the numeration in each string
    all_preds, all_true = get_all_preds_and_true("{}/pred_true.txt", losses)

    # All are same so get the first
    all_true = all_true[0]
    for image_num in range(0, 50):
        data = ""
        aux = []
        for i, _ in enumerate(losses):

            cl_pred = all_preds[i][image_num]

            if cl_pred in aux:
                continue
            else:
                data += column.format(image_num, cl_pred)
                aux.append(cl_pred)

        if image_num == 12:
            page += bait.format('circle', url, image_num)
        elif image_num == 42:
            page += bait.format('triangle', url, image_num)

        page += item_holder.format(image_num, url, data, all_true[image_num])

    page += footer
    savepath = os.path.join(os.getcwd(), "study.html")
    print("[+]: Saving HTML in {}".format(savepath))
    html_page = open(savepath, "w")
    html_page.write(page)


def save_within_subject_mturk_html_page_cifar10_extra(url, study_name):
    """
    Within-Subjects design of the MTurk studies: Each subject is asked to give their opinion on each classifier
    :param url: Base URL where the images are going to be placed. Need to be accessed via Internet
    :return: Writes a new HTML suitable for the MTUrk studies
    """

    header = '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />\
    <section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">\
    <div class="row col-xs-12 col-md-12"><!-- Instructions -->\
    <div class="panel panel-primary">\
    <div class="panel-heading"><strong>Instructions</strong></div>\
    <div class="panel-body">\
    <p>As a part of this HIT, you will be asked to answer questions about 62 images. For each image, select the most appropriate option.</p>\
    <p>This study is part of a research effort. We request you to answer the questions sincerely.</p>\
    <p>** YOU MUST ANSWER ALL QUESTIONS TO GET YOUR HIT APPROVED (TO RECEIVE PAYMENT). **</p>\
    </div>\
    </div>\
    <!-- End Instructions --><!-- Content Body -->\
    <section>\n\n'

    item_holder = '<div><img height="100" width="100" src="{1}/{0}" style="display: block; margin-left: auto; margin-right: auto;"/>\
        <h5><b>Q{4}:</b> The above image represents a <b>{3}</b>. How reasonable are the following misclassifications?</h5>\
        <table>\
            <tr>\
                {2}\
            </tr>\
        </table>\
        </div>'

    column = '<td>\
                <fieldset><label><span style="font-weight:normal"> Classifier\'s prediction: </span> {1} </label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="0" />Highly unreasonable (surprised)</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="1" />Somewhat unreasonable</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="2" />Not sure</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="3" />Somewhat reasonable</div></label>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="4" />Highly reasonable (understandable)</div></label>\
                    <hr>\
                    <div class="radio"><label><input name="img_{0}_pred_{1}" type="radio" value="-1" />Cannot tell from the picture</div></label>\
    </fieldset>\n\n\
            </td>'

    bait = '<div><img height="200" width="200" src="{1}/{0}.jpg" style="display: block; margin-left: auto; margin-right: auto;"/></div>\
    <fieldset><label>This is a: </label>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="circle" />Circle</div>\
    <div class="radio"><input name="similarity_{0}_{2}" type="radio" value="triangle" />Triangle</div>\
    </fieldset>\n\n'

    footer = '</section>\
    <!-- End Content Body --></div>\
    </section>\
    <style type="text/css">fieldset {padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }\
    </style>\n'

    def get_cifar10_extra_preds_true(n):
        preds_l, base_l, true_l, ids_l, img_name_l = [], [], [], [], []
        path = "./human_studies/cifar10_extra/egregious_examples/"
        path_selected = os.path.join(path, "selected")
        image_names = os.listdir(path)
        random.shuffle(image_names)

        count = 0
        for image_name in image_names:
            # The images we want start with their respective number
            if not image_name[0].isdigit(): continue
            img_name_l.append(image_name)
            preds_set = set()
            image_name = image_name.split(".")[0]

            id, true, base, ihl, chl, ekl = image_name.split("-")

            ids_l.append(id)
            true_l.append(true.split("_")[1])
            base_l.append(base.split("_")[1])
            preds_set.add(ihl.split("_")[1])
            preds_set.add(chl.split("_")[1])
            preds_set.add(ekl.split("_")[1])
            preds_l.append(list(preds_set))

            count += 1
            if count == n: break

        print("[+]: Copying selected random images")
        for img_name in img_name_l:
            shutil.copyfile(os.path.join(path, img_name), os.path.join(path_selected, img_name))

        return true_l, base_l, preds_l, ids_l, img_name_l

    # Important note: To generate the images first run instance_based_comparison.py
    page = header
    true_l, base_l, preds_l, ids_l, names_l = get_cifar10_extra_preds_true(60)

    for image_num, (true, base, preds, id, fname) in enumerate(zip(true_l, base_l, preds_l, ids_l, names_l)):
        data = ""

        data += column.format(id, base)
        for wlf in preds:
            data += column.format(id, wlf)

        if image_num == 12:
            page += bait.format('circle', url, image_num)
        elif image_num == 42:
            page += bait.format('triangle', url, image_num)

        page += item_holder.format(fname, url, data, true, image_num + 1)

    page += footer
    savepath = os.path.join(os.getcwd(), "human_studies", study_name)
    print("[+]: Saving HTML in {}".format(savepath))
    html_page = open(savepath, "w")
    html_page.write(page)


def get_random_miscl_imgs():
    for loss in losses:
        y_pred = np.argmax(np.load(y_pred_file.format(loss)), 1)
    y_corr = y_pred - y_test
    y_miscl = np.nonzero(y_corr)[0]
    selected = np.random.choice(y_miscl, N)

    print("[+]: Saving images for {}".format(loss))
    for i, img in enumerate(x_test[selected]):
        img = Image.fromarray(img)
        img.save("{}/{}.jpg".format(loss, i))
    print("[+]: Saving labels for {}".format(loss))

    file = open(classnames_file.format(loss), "w")
    for y_pr, y_te in zip(y_pred[selected], y_test[selected]):
        line = "{},{}\n".format(c10_classnames[y_pr], c10_classnames[y_te])
        file.write(line)


def process_cifar10_mturk_userstudy(filename):
    prefix = ""
    filename = prefix + filename
    df = pd.read_csv(filename)
    # Get the data columns
    df = df[df.columns[27:-2]]
    df = df[df["Answer.similarity_triangle_42"] == "triangle"]
    df = df[df["Answer.similarity_circle_12"] == "circle"]
    df = df[df.columns[:-2]]

    preds_dict = {}
    for loss in losses:
        path = prefix + "preds/pred_true_{}.txt".format(loss)
        pred, _ = get_preds_and_true_labels(path)
        preds_dict[loss] = pred
        preds_dict[loss + "_score"] = []

    for col in df:
        img_num, pred = int(col.split("_")[1]), col.split("_")[-1]
        likert_score = df[col].mean()
        for loss in losses:
            if preds_dict[loss][img_num] == pred:
                preds_dict[loss + "_score"].append(likert_score)

    for loss in losses:
        preds_dict[loss + "_score"] = mean(preds_dict[loss + "_score"])
        print("[+]: Average {} Likert score is {}".format(loss, preds_dict[loss + "_score"]))


# TODO: Save IDs in separate file
def process_cifar10_extra_mturk_study(filename):
    pd.set_option('display.max_columns', None)

    prefix = "human_studies/cifar10_extra"
    filename = os.path.join(prefix, filename)
    df = pd.read_csv(filename)
    # Get the data columns
    df = df[df.columns[27:-2]]
    df = df[df["Answer.similarity_triangle_42"] == "triangle"]
    df = df[df["Answer.similarity_circle_12"] == "circle"]
    df = df[df.columns[:-2]]

    # NOTE: Make sure images are ordered by their number in both the CSV and the folder
    # TODO: Treat -1
    df.replace(-1, np.NaN)
    path_imgs = os.path.join(prefix, "egregious_examples/")
    data = {"base": [], "IHL": [], "CHL": [], "EKL": []}
    models_name = list(data.keys())

    selected_ids = set([x.split(".")[1].split("_")[1] for x in df.columns])
    all_images = [x for x in sorted(os.listdir(path_imgs)) if x[0].isdigit()]
    for img_name in all_images:
        id_true_preds = img_name.split(".")[0].split("-")
        id, true, models_pred = id_true_preds[0], id_true_preds[1], id_true_preds[2:]
        if id not in selected_ids: continue
        cur_df = df.filter(like=id).mean(skipna=True)
        for name, pred in zip(models_name, models_pred):
            pred = pred.split("_")[1]
            # print(f"{name}, {pred}, {cur_df.filter(like=pred).values[0]}")
            data[name].append(cur_df.filter(like=pred).values[0])

    for name in data:
        data[name] = mean(data[name])
        print(f"{name}: {data[name]}")

    print(f"Total turkers: {len(df)}")


##########################################################################################
##########################################################################################


# TODO: Note -- run with terminal on the server
if __name__ == '__main__':
    # ImageNet user study
    # ------------------
    # save_grid_and_images_for_userstudy_imagenet()
    # save_within_subject_mturk_html_page_imagenet("https://aolmo.github.io/assets/web/exps/miscl-imagenet/")
    # save_within_subject_mturk_html_page_imagenet("https://www.public.asu.edu/~ssengu15/exp_data/imgnt5344477688078653/")
    # f = "./imagenet/within_study_random_imagenet_54.csv"
    # process_imagenet_mturk_userstudy("./imagenet/within_study_egregious_imagenet_54.csv")

    # Cifar10 user study
    # ------------------
    # get_same_miscl_imgs()
    # save_within_subject_mturk_html_page("https://www.public.asu.edu/~ssengu15/exp_data/{}/")
    # save_between_subject_mturk_html_page("https://www.public.asu.edu/~ssengu15/exp_data/wlf7842320382223565")

    # Process data extracted from the study
    # process_cifar10_mturk_userstudy("./human_studies/run1_50_data.csv")

    # Cifar10 Extra user study
    # ------------------
    # save_within_subject_mturk_html_page_cifar10_extra("https://aolmo.github.io/assets/web/exps/miscl-cifar10-extra", "cifar-10-extra-mturk.html")
    process_cifar10_extra_mturk_study("cifar10_extra_mturk_data.csv")
