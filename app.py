from __future__ import print_function, division
import json
import csv
import os
import glob
import random
from flask import Flask, render_template, request, redirect, Response, jsonify, make_response
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import seaborn as sns
import matplotlib.patches as patches
from PIL import Image
import webcolors
import os,io
import pandas as pd
import cv2
from google.cloud import vision
from google.cloud.vision import types
from pandas.io.json import json_normalize


from collections import OrderedDict
try:
    import pickle
except ImportError:
    import cPickle as pickle  # pylint: disable=import-error
from pkg_resources import parse_version

from PIL import Image
import numpy as np
assert parse_version(np.__version__) >= parse_version('1.9.0'), \
    "numpy >= 1.9.0 is required for daltonize"
try:
    import matplotlib as mpl
    _NO_MPL = False
except ImportError:
    _NO_MPL = True
sns.set(style="ticks")

#New Code
#!/usr/bin/env python

"""
   Written by Joerg Dietrich <astro@joergdietrich.com>. Copyright 2015
   Based on original code by Oliver Siemoneit. Copyright 2007
   This code is licensed under the GNU GPL version 2, see COPYING for details.
"""




def transform_colorspace(img, mat):
    """Transform image to a different color space.

    Arguments:
    ----------
    img : array of shape (M, N, 3)
    mat : array of shape (3, 3)
        conversion matrix to different color space

    Returns:
    --------
    out : array of shape (M, N, 3)
    """
    # Fast element (=pixel) wise matrix multiplication
    return np.einsum("ij, ...j", mat, img)


def simulate(img, color_deficit="d"):
    """Simulate the effect of color blindness on an image.

    Arguments:
    ----------
    img : PIL.PngImagePlugin.PngImageFile, input image
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia

    Returns:
    --------
    sim_rgb : array of shape (M, N, 3)
        simulated image in RGB format
    """
    # Colorspace transformation matrices
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01, 1.16721066e-01],
                        [-1.02485335e-02, 5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03, 6.93511405e-01]])

    img = img.copy()
    img = img.convert('RGB')

    rgb = np.asarray(img, dtype=float)
    # first go from RBG to LMS space
    lms = transform_colorspace(rgb, rgb2lms)
    # Calculate image as seen by the color blind
    sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
    # Transform back to RBG
    sim_rgb = transform_colorspace(sim_lms, lms2rgb)
    return sim_rgb


def daltonize(rgb, color_deficit='d'):
    """
    Adjust color palette of an image to compensate color blindness.

    Arguments:
    ----------
    rgb : array of shape (M, N, 3)
        original image in RGB format
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia

    Returns:
    --------
    dtpn : array of shape (M, N, 3)
        image in RGB format with colors adjusted
    """
    sim_rgb = simulate(rgb, color_deficit)
    err2mod = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    # rgb - sim_rgb contains the color information that dichromats
    # cannot see. err2mod rotates this to a part of the spectrum that
    # they can see.
    rgb = rgb.convert('RGB')
    err = transform_colorspace(rgb - sim_rgb, err2mod)
    dtpn = err + rgb
    return dtpn


def array_to_img(arr):
    """Convert a numpy array to a PIL image.

    Arguments:
    ----------
    arr : array of shape (M, N, 3)

    Returns:
    --------
    img : PIL.Image.Image
        RGB image created from array
    """
    # clip values to lie in the range [0, 255]
    arr = clip_array(arr)
    arr = arr.astype('uint8')
    img = Image.fromarray(arr, mode='RGB')
    return img


def clip_array(arr, min_value=0, max_value=255):
    """Ensure that all values in an array are between min and max values.

    Arguments:
    ----------
    arr : array_like
    min_value : float, optional
        default 0
    max_value : float, optional
        default 255

    Returns:
    --------
    arr : array_like
        clipped such that all values are min_value <= arr <= max_value
    """
    comp_arr = np.ones_like(arr)
    arr = np.maximum(comp_arr * min_value, arr)
    arr = np.minimum(comp_arr * max_value, arr)
    return arr


def get_child_colors(child, mpl_colors):
    """
    Recursively enter all colors of a matplotlib objects and its
    children into a dictionary.

    Arguments:
    ----------
    child : a matplotlib object
    mpl_colors : OrderedDict from collections

    Returns:
    --------
    mpl_colors : OrderedDict
    """
    mpl_colors[child] = OrderedDict()
    if hasattr(child, "get_color"):
        mpl_colors[child]['color'] = child.get_color()
    if hasattr(child, "get_facecolor"):
        mpl_colors[child]['fc'] = child.get_facecolor()
    if hasattr(child, "get_edgecolor"):
        mpl_colors[child]['ec'] = child.get_edgecolor()
    if hasattr(child, "get_markeredgecolor"):
        mpl_colors[child]['mec'] = child.get_markeredgecolor()
    if hasattr(child, "get_markerfacecolor"):
        mpl_colors[child]['mfc'] = child.get_markerfacecolor()
    if hasattr(child, "get_markerfacecoloralt"):
        mpl_colors[child]['mfcalt'] = child.get_markerfacecoloralt()
    if isinstance(child, mpl.image.AxesImage):
        mpl_colors[child]['cmap'] = child.get_cmap()
        img_properties = child.properties()
        try:
            img_arr = img_properties['array']
            if len(img_arr.shape) == 3:
                mpl_colors[child]['array'] = np.array(img_arr)
        except KeyError:
            pass
    if hasattr(child, "get_children"):
        grandchildren = child.get_children()
        for grandchild in grandchildren:
            mpl_colors = get_child_colors(grandchild, mpl_colors)
    return mpl_colors


def get_mpl_colors(fig):
    """
    Read all colors used in a matplotlib figure into an OrderedDict.

    Arguments:
    ----------
    fig : matplotlib.figure.Figure

    Returns:
    --------
    mpl_dict : OrderedDict from collections
    """
    mpl_colors = OrderedDict()
    children = fig.get_children()
    for child in children:
        mpl_colors = get_child_colors(child, mpl_colors)
    return mpl_colors


def get_key_colors(mpl_colors, rgb, alpha):
    """From an OrderedDict of colors of all figure object children
    recursively fill rgb and alpha channel information.

    Arguments:
    ----------
    mpl_colors : OrderedDict
        dictionary with all colors of all children, matplotlib instances are
        keys
    rgb : array of shape (M, 1, 3)
        line image holding RGB colors encountered so far.
    alpha : array of shape (M, 1)
        line image holding alpha values encountered so far.

    Returns:
    --------
    rgb : array of shape (M+n, 1, 3)
    alpha : array of shape (M+n, 1)
    """
    if _NO_MPL is True:
        raise ImportError("matplotlib not found, "
                          "can only deal with pixel images")
    cc = mpl.colors.ColorConverter()  # pylint: disable=invalid-name
    # Note that the order must match the insertion order in
    # get_child_colors()
    color_keys = ("color", "fc", "ec", "mec", "mfc", "mfcalt", "cmap", "array")
    for color_key in color_keys:
        try:
            color = mpl_colors[color_key]
            # skip unset colors, otherwise they are turned into black.
            if isinstance(color, str) and color == 'none':
                continue
            if isinstance(color, mpl.colors.LinearSegmentedColormap):
                rgba = color(np.arange(color.N))
            elif isinstance(color, np.ndarray) and color_key == "array":
                color = color.reshape(-1, 3) / 255
                a = np.zeros((color.shape[0], 1))  # pylint: disable=invalid-name
                rgba = np.hstack((color, a))
            else:
                rgba = cc.to_rgba_array(color)
            rgb = np.append(rgb, rgba[:, :3])
            alpha = np.append(alpha, rgba[:, 3])
        except KeyError:
            pass
        for key in mpl_colors.keys():
            if key in color_keys:
                continue
            rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    return rgb, alpha


def arrays_from_dict(mpl_colors):
    """
    Create rgb and alpha arrays from color dictionary.

    Arguments:
    ----------
    mpl_colors : OrderedDict
        dictionary with all colors of all children, matplotlib instances are
        keys

    Returns:
    --------
    rgb : array of shape (M, 1, 3)
        RGB values of colors in a line image, M is the total number of
        non-unique colors
    alpha : array of shape (M, 1)
        Alpha channel values of all mpl instances
    """
    rgb = np.array([])
    alpha = np.array([])
    for key in mpl_colors.keys():
        rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    m = rgb.size // 3  # pylint: disable=invalid-name
    rgb = rgb.reshape((m, 1, 3))
    return rgb, alpha


def _set_colors_from_array(instance, mpl_colors, rgba, i=0):
    """
    Set object instance colors to the modified ones in rgba.
    """
    cc = mpl.colors.ColorConverter()  # pylint: disable=invalid-name
    # Note that the order must match the insertion order in
    # get_child_colors()
    color_keys = ("color", "fc", "ec", "mec", "mfc", "mfcalt", "cmap", "array")
    for color_key in color_keys:
        try:
            color = mpl_colors[color_key]
            if isinstance(color, mpl.colors.LinearSegmentedColormap):
                j = color.N
            elif isinstance(color, np.ndarray) and color_key == "array":
                j = color.shape[0] * color.shape[1]
            else:
                # skip unset colors, otherwise they are turned into black.
                if isinstance(color, str) and color == 'none':
                    continue
                color_shape = cc.to_rgba_array(color).shape
                j = color_shape[0]
            target_color = rgba[i: i + j, :]
            if j == 1:
                target_color = target_color[0]
            i += j
            if color_key == "color":
                instance.set_color(target_color)
            elif color_key == "fc":
                instance.set_facecolor(target_color)
            elif color_key == "ec":
                instance.set_edgecolor(target_color)
            elif color_key == "mec":
                instance.set_markeredgecolor(target_color)
            elif color_key == "mfc":
                instance.set_markerfacecolor(target_color)
            elif color_key == "mfcalt":
                instance.set_markerfacecoloralt(target_color)
            elif color_key == "cmap":
                instance.set_cmap(
                    instance.cmap.from_list(instance.cmap.name+"_dlt",
                                            target_color))
            elif color_key == "array":
                target_color = (target_color.reshape((color.shape[0],
                                                      color.shape[1],
                                                      -1)))
                target_color = (target_color[:, :, :3] * 255).astype('uint8')
                instance.set_data(target_color)
        except KeyError:
            pass
    return i


def set_mpl_colors(mpl_colors, rgba):
    """
    Recursively set the colors in a color dictionary to new values in rgba.

    Arguments:
    ----------
    mpl_colors : OrderedDict
        dictionary with all colors of all children, matplotlib instances are
        keys

    rgba : array of shape (M, 1, 4) containing rgb, alpha channels
    """
    i = 0
    for key in mpl_colors.keys():
        i = _set_colors_from_array(key, mpl_colors[key], rgba, i)


def _prepare_for_transform(fig):
    """
    Gather color keys/info for mpl figure and arange them such that the image
    simulate() or daltonize() routines can be called on them.
    """
    mpl_colors = get_mpl_colors(fig)
    rgb, alpha = arrays_from_dict(mpl_colors)
    return rgb, alpha, mpl_colors


def _join_rgb_alpha(rgb, alpha):
    """
    Combine (m, n, 3) rgb and (m, n) alpha array into (m, n, 4) rgba.
    """
    rgb = clip_array(rgb, 0, 1)
    r, g, b = np.split(rgb, 3, 2)  # pylint: disable=invalid-name, unbalanced-tuple-unpacking
    rgba = np.concatenate((r, g, b, alpha.reshape(alpha.size, 1, 1)),
                          axis=2).reshape(-1, 4)
    return rgba


def simulate_mpl(fig, color_deficit='d', copy=False):
    """
    Simulate color blindness on a matplotlib figure.

    Arguments:
    ----------
    fig : matplotlib.figure.Figure
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    copy : bool, optional
        should simulation happen on a copy (True) or the original
        (False, default)

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    if copy:
        # mpl.transforms cannot be copy.deepcopy()ed. Thus we resort
        # to pickling.
        pfig = pickle.dumps(fig)
        fig = pickle.loads(pfig)
    rgb, alpha, mpl_colors = _prepare_for_transform(fig)
    sim_rgb = simulate(array_to_img(rgb * 255), color_deficit) / 255
    rgba = _join_rgb_alpha(sim_rgb, alpha)
    set_mpl_colors(mpl_colors, rgba)
    fig.canvas.draw()
    return fig


def daltonize_mpl(fig, color_deficit='d', copy=False):
    """
    Daltonize a matplotlib figure.

    Arguments:
    ----------
    fig : matplotlib.figure.Figure
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    copy : bool, optional
        should daltonization happen on a copy (True) or the original
        (False, default)

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    if copy:
        # mpl.transforms cannot be copy.deepcopy()ed. Thus we resort
        # to pickling.
        pfig = pickle.dumps(fig)
        fig = pickle.loads(pfig)
    rgb, alpha, mpl_colors = _prepare_for_transform(fig)
    dtpn = daltonize(array_to_img(rgb * 255), color_deficit) / 255
    rgba = _join_rgb_alpha(dtpn, alpha)
    set_mpl_colors(mpl_colors, rgba)
    fig.canvas.draw()
    return fig


# if __name__ == '__main__':
#     import argparse

#     # pylint: disable=invalid-name
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_image", type=str)
#     parser.add_argument("output_image", type=str)
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument("-s", "--simulate", help="create simulated image",
#                        action="store_true")
#     group.add_argument("-d", "--daltonize",
#                        help="adjust image color palette for color blindness",
#                        action="store_true")
#     parser.add_argument("-t", "--type", type=str, choices=["d", "p", "t"],
#                         help="type of color blindness (deuteranopia, "
#                         "protanopia, tritanopia), default is deuteranopia "
#                         "(most common)")
#     args = parser.parse_args()

#     if args.simulate is False and args.daltonize is False:
#         print("No action specified, assume daltonizing")
#         args.daltonize = True
#     if args.type is None:
#         args.type = "d"

#     orig_img = Image.open(args.input_image)

#     if args.simulate:
#         simul_rgb = simulate(orig_img, args.type)
#         simul_img = array_to_img(simul_rgb)
#         simul_img.save(args.output_image)
#     if args.daltonize:
#         dalton_rgb = daltonize(orig_img, args.type)
#         dalton_img = array_to_img(dalton_rgb)
#         dalton_img.save(args.output_image)



#Old Code

app = Flask(__name__)
randomSampleCount = int(0.25 * 1000)
feature_selected = ''
categorical_features = ['Type','Year','City','Rating','Major Size','Population']
numerical_features = ['Price','Total Volume','Total Bags','PLU4046','PLU4225','PLU4770','Small Bags','Large Bags','XLarge Bags']
stratified_sample = pd.DataFrame()
random_sample = pd.DataFrame()
eigen_values_org = []
eigen_values_random = []
eigen_values_stratified = []  
eigen_vectors_org = []
eigen_vectors_random = []
eigen_vectors_stratified = []
x_ticks_sqLoad_org = []
x_ticks_sqLoad_random = []
x_ticks_sqLoad_strat = []
squared_loadings_org = []
squared_loadings_random = []
squared_loadings_strat = []
colors=["red","green","blue","pink","yellow"]

y_ticks_org = []
y_ticks_random = []
y_ticks_stratified = []
clustered_sample = []

df_chart=pd.DataFrame(columns=['Chart Labels','Prediction Status'])


@app.route("/")
def index():

    return render_template("AdvancedProject.html")

@app.route("/t3", methods=['POST'])
def t3():
    PEOPLE_FOLDER = os.path.join('static', 'academic')
    app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
    # if request.method == 'POST':  
    #     f = request.files['file']  
    #     f.save(f.filename)  
    #     os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccountToken.json'
    #     client = vision.ImageAnnotatorClient()
    #     global image_path
    #     image_path=f.filename

    #     # img=cv2.imread(image_path)
    #     orig_img = Image.open(image_path) 
    #     dalton_rgb = daltonize(orig_img, 't')
    #     dalton_img = array_to_img(dalton_rgb)
    #     dalton_img.save(f.filename)
    #     plt.imshow(dalton_img)

    #     fig=plt.gcf()
    #     plt.axis('off')
    r='out2'+'.jpg'
    # r=static'+'_'+'1'+'.png'
    #     fig.savefig(p)
        # fig.clear()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], r)

    return render_template("PredictedLabels.html",user_image=full_filename)

@app.route("/yes", methods=['POST'])
def yes():
    PEOPLE_FOLDER = os.path.join('static', 'academic')
    app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
    # if request.method == 'POST':  
    #     f = request.files['file']  
    #     f.save(f.filename)  
    #     os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccountToken.json'
    #     client = vision.ImageAnnotatorClient()
    #     global image_path
    #     image_path=f.filename

    #     # img=cv2.imread(image_path)
    #     orig_img = Image.open(image_path) 
    #     dalton_rgb = daltonize(orig_img, 't')
    #     dalton_img = array_to_img(dalton_rgb)
    #     dalton_img.save(f.filename)
    #     plt.imshow(dalton_img)

    #     fig=plt.gcf()
    #     plt.axis('off')
    r='out1'+'.jpg'
    # r=static'+'_'+'1'+'.png'
    #     fig.savefig(p)
        # fig.clear()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], r)

    return render_template("PredictedLabels.html",user_image=full_filename)


@app.route('/success', methods = ['POST'])  
def success(): 

    PEOPLE_FOLDER = os.path.join('static', 'academic')
    app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccountToken.json'
        client = vision.ImageAnnotatorClient()
        global image_path
        image_path=f.filename

        # img=cv2.imread(image_path)
        orig_img = Image.open(image_path) 
        dalton_rgb = daltonize(orig_img, 'd')
        dalton_img = array_to_img(dalton_rgb)
        dalton_img.save(f.filename)
        plt.imshow(dalton_img)

        # def Bounding_Box(response,fh,fw):
            
        #     df1 = pd.DataFrame(columns=['Text', 'xp', 'yp','x2p','y2p','xcp','ycp','wp','hp'])
        #     i=0
        #     for text in response.text_annotations:
        #         if "\n" in text.description:
        #             continue

        #         j=0
        #         for v in text.bounding_poly.vertices:
        #             if j==0:
        #                 bottom_left_x=v.x
        #                 bottom_left_y=v.y
        #             if j==1:
        #                 right_bottom_x=v.x
        #                 right_bottom_y=v.y
        #             if j==2:
        #                 top_right_x=v.x
        #                 top_right_y=v.y
        #             if j==3:
        #                 top_left_x=v.x
        #                 top_left_y=v.y
        #             j+=1

                

        #         #Top-left and Bottom-right Coordinates
        #         xl=top_left_x
        #         yl=top_left_y
        #         xr=right_bottom_x
        #         yr=right_bottom_y

        #         #Center Coordinates
        #         xc=(xl+yr)/2.0
        #         yc=(yl+yr)/2.0

        #         #Dimension of Bounding-Box
        #         w=abs(xr-xl)
        #         h=abs(yr-yl)

        #         #Normalize the coordinates
        #         xl/=fw
        #         yl/=fh
        #         xr/=fw
        #         yr/=fh
        #         xc/=fw
        #         yc/=fh
        #         w/=fw
        #         h/=fh


        #         # im=cv2.imread(image_path)
        #         # plt.imshow(im)

        #         # ax = plt.gca()

        #         # rect = patches.Rectangle((bottom_left_x,bottom_left_y),w,h,linewidth=2,edgecolor='green',fill = False)

        #         # ax.add_patch(rect)


        #         #Push in Dataframe
        #         df1.loc[i] = [text.description] + [xl,yl,xr,yr,xc,yc,w,h]
        #         i+=1

                

        #     # fig=plt.gcf()
        #     # plt.axis('off')
        #     # img2=plt.show()
        #     # fig.savefig('static/image/new_plot.png')
                
        #     return df1

        # def test_feature(file_path):
        #     image_path =file_path

        #     #To get Image Shape
        #     img=cv2.imread(image_path)
        #     fh,fw,c=img.shape

        #     #Open file from path
        #     with io.open(image_path,'rb') as image_file:
        #         content = image_file.read()

        #     # construct an image instance
        #     image = vision.types.Image(content=content)

        #     # annotate Image Response : this would be in JSON format
        #     global response
        #     response = client.text_detection(image=image)  # returns TextAnnotation
        #     df = pd.DataFrame(columns=['locale', 'description'])

        #     texts = response.text_annotations

        #     for text in texts:
        #         df = df.append(
        #             dict(
        #                 locale=text.locale,
        #                 description=text.description
        #             ),
        #             ignore_index=True
        #         )
        #         v=text.bounding_poly.vertices

        #     #Coordinates will be a Dataframe
        #     Coordinates=Bounding_Box(response,fh,fw)
        #     return Coordinates
            
        # file = open('svm_model.pkl', 'rb')
        # final_model = pickle.load(file)
        # file.close()

        # file = open('svm_scaler.pkl', 'rb')
        # scaler = pickle.load(file)
        # file.close()

        # file = open('svm_encoder.pkl', 'rb')
        # encoder = pickle.load(file)
        # file.close()


        def get_path(image_path):
            df1 = pd.DataFrame(columns=['Text', 'xp', 'yp','x2p','y2p','xcp','ycp','wp','hp'])
            i=0
            for text in response.text_annotations:
                if "\n" in text.description:
                    continue

                j=0
                for v in text.bounding_poly.vertices:
                    if j==0:
                        bottom_left_x=v.x
                        bottom_left_y=v.y
                    if j==1:
                        right_bottom_x=v.x
                        right_bottom_y=v.y
                    if j==2:
                        top_right_x=v.x
                        top_right_y=v.y
                    if j==3:
                        top_left_x=v.x
                        top_left_y=v.y
                    j+=1

                

                #Top-left and Bottom-right Coordinates
                xl=top_left_x
                yl=top_left_y
                xr=right_bottom_x
                yr=right_bottom_y


                #Dimension of Bounding-Box
                w=abs(xr-xl)
                h=abs(yr-yl)


                im=cv2.imread(image_path)
                plt.imshow(im)

                ax = plt.gca()

                rect = patches.Rectangle((bottom_left_x,bottom_left_y),w,h,linewidth=2,edgecolor='green',fill = False)

                ax.add_patch(rect)


                #Push in Dataframe
                # df1.loc[i] = [text.description] + [xl,yl,xr,yr,xc,yc,w,h]
                # i+=1
                

            # fig=plt.gcf()
            # plt.axis('off')
            # # img2=plt.show()
            # p='static'+'/'+'academic'+'/'+image_path+'_'+'1'+'.png'
            # r=image_path+'_'+'1'+'.png'
            # fig.savefig(p)
            # fig.clear()
            return r


        def chart_labels(image_path):
            x_test=test_feature(image_path)
            myfeat= ['xp', 'yp','x2p','y2p','xcp','ycp','wp','hp']
            x_test = x_test[myfeat]

            x_test_scaled = scaler.transform(x_test)

            y_test_pred = final_model.predict(x_test_scaled)
            y_test_pred_label = list(encoder.inverse_transform(y_test_pred))
            return y_test_pred_label
        


        # path = "Images"
        # images = os.listdir(path)
        # images.sort()
        # img_items=[]
        # for img in images:
        #     if "jpg" not in img: 
        #         continue
        #     img_items= img_items+ [img]


            
        # try:
        #     test_pred_labels= chart_labels(image_path)
        #     test_pred_labels= np.unique(test_pred_labels)
        #     print('\033[1m' + 'Predicted Chart Labels are: ' + str(test_pred_labels))
        #     print(type(test_pred_labels))
        #     df = pd.DataFrame(data=test_pred_labels, columns=["Predicted Labels"])
        #     print(df)
        # except :
        #     print('\033[1m' + "OOPS !!: Very Poor Chart no Labels Found !")
        
        
        # global df_chart
        # a=[]
        # b=['x-axis-label','x-axis-title','y-axis-label','y-axis-title','legend-label','chart-title']
        # for i in range(len(df)):
        #     a.append(df['Predicted Labels'].iloc[i])
        # print(b)
        # z=0
        # for i in b:
        #     if i in a:
        #         df_chart.loc[z]=[i]+[u'\u2705']
        #     else:
        #         df_chart.loc[z]=[i]+[u'\u274c']
        #     z+=1

        # new_image_path = get_path(image_path)
        fig=plt.gcf()
        plt.axis('off')
        p='static'+'/'+'academic'+'/'+image_path+'_'+'1'+'.png'
        r='out0'+'.jpg'
        fig.savefig(p)
        fig.clear()
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], r)
        # print(new_image_path)
        # print("here",df)
        # print("new",df_chart)
        print(full_filename)
        return render_template("PredictedLabels.html",user_image=full_filename)
        # return render_template("AdvancedProject.html", name = f.filename)  


@app.route('/Labels', methods = ['POST','GET'])  
def Labels(): 
    global df_chart
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
    print(image_path)
    print(df_chart)
    return render_template("Labels.html",tables=[df_chart.to_html(classes='data')], titles=df_chart.columns.values, user_image=full_filename)

@app.route("/dashboard")
def dashboard():
    return render_template("visual.html")

@app.route("/AdvancedProject", methods = ['POST', 'GET'])
def AdvancedProject():
   
    return render_template("AdvancedProject.html")

@app.route("/Coordinates", methods = ['POST', 'GET'])
def Coordinates():
    # if request.method == 'POST':
    # f = request.files['file']
    # image_path =f.filename
    # print('Inside Post')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccountToken.json'
    client = vision.ImageAnnotatorClient()

    file_name = 'test.jpg'
    # image_path='Images/test.jpg'

    #To get Image Shape
    img=cv2.imread(image_path)
    fh,fw,c=img.shape

    #Open file from path
    with io.open(image_path,'rb') as image_file:
        content = image_file.read()

    # construct an image instance
    image = vision.types.Image(content=content)

    # annotate Image Response : this would be in JSON format
    response = client.text_detection(image=image)  # returns TextAnnotation
    df = pd.DataFrame(columns=['locale', 'description'])

    texts = response.text_annotations

    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )
        v=text.bounding_poly.vertices


    def Bounding_Box(response):
        df1 = pd.DataFrame(columns=['Text', 'xp', 'yp','x2p','y2p','xcp','ycp','wp','hp'])

        i=0
        for text in response.text_annotations:
            if "\n" in text.description:
                continue

            j=0
            for v in text.bounding_poly.vertices:
                if j==1:
                    right_bottom_x=v.x
                    right_bottom_y=v.y
                if j==3:
                    top_left_x=v.x
                    top_left_y=v.y
                j+=1

            #Top-left and Bottom-right Coordinates
            xl=top_left_x
            yl=top_left_y
            xr=right_bottom_x
            yr=right_bottom_y

            #Center Coordinates
            xc=(xl+yr)/2.0
            yc=(yl+yr)/2.0

            #Dimension of Bounding-Box
            w=abs(xr-xl)
            h=abs(yr-yl)

            #Normalize the coordinates
            xl/=fw
            yl/=fh
            xr/=fw
            yr/=fh
            xc/=fw
            yc/=fh
            w/=fw
            h/=fh

            #Push in Dataframe
            df1.loc[i] = [text.description] + [xl,yl,xr,yr,xc,yc,w,h]
            i+=1
            
        
        return df1
        
    #Coordinates will be a Dataframe
    Coordinates=Bounding_Box(response)

    print(Coordinates)

    return render_template("Coordinates.html",tables=[Coordinates.to_html(classes='data')], titles=Coordinates.columns.values)

@app.route("/Color", methods = ['POST', 'GET'])
def Color():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccountToken.json'

    with io.open(image_path,'rb') as image_file:
        content = image_file.read()

    client = vision.ImageAnnotatorClient()

    image = vision.types.Image(content=content)
    response = client.image_properties(image=image).image_properties_annotation
    dominant_colors = response.dominant_colors

    a=[]
    b=[]
    c=[]
    d=[]
    i=0
    for color in dominant_colors.colors:
        a.append(color.pixel_fraction)
        b.append(color.color.red)
        c.append(color.color.green)
        d.append(color.color.blue)


    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]



    df1=pd.DataFrame(a,columns=['Pixel Fraction'])
    df2=pd.DataFrame(b,columns=['Red'])
    df3=pd.DataFrame(c,columns=['Green'])
    df4=pd.DataFrame(d,columns=['Blue'])

    y=df1.merge(df2,left_index=True, right_index=True)
    z=y.merge(df3,left_index=True, right_index=True)
    x=z.merge(df4,left_index=True, right_index=True)

    actual=[]
    closest=[]
    for i in range(len(x)):
        requested_colour=(x['Red'].iloc[i],x['Green'].iloc[i],x['Blue'].iloc[i])
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = closest_colour(requested_colour)
            actual_name = None
        actual.append(actual_name)
        closest.append(closest_name)

        print("Actual colour name:", actual_name, ", closest colour name:", closest_name)

    df5=pd.DataFrame(actual,columns=['Actual_Color'])
    df6=pd.DataFrame(closest,columns=['Closest_Color'])

    actualdf=x.merge(df5,left_index=True, right_index=True)
    closestdf=actualdf.merge(df6,left_index=True, right_index=True)

    # fig=plt.gcf()
    # plt.pie(df1, labels=closest, colors=closest, autopct='%1.1f%%', shadow=True, startangle=90)
    # # plt.show()
    # pa='static'+'/'+'academic'+'/'+image_path+'_'+'2'+'.png'
    # fig.savefig(pa)
    # fig.clear()

    # df=pd.DataFrame(a,columns=['Pixel Fraction'])
    # dfa=pd.DataFrame(closest,columns=['Closest_Color'])
    # dfb=dfa.merge(df,left_index=True, right_index=True)
    # # dfb.plot(kind='bar', stacked=True, color=closest)
    
    # customPalette = sns.set_palette(sns.color_palette(closest))
    # ax = sns.barplot(x="Closest_Color", y="Pixel Fraction", hue="Closest_Color", data=dfb, palette=customPalette)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    # fig = ax.get_figure()
    # pa='static'+'/'+'academic'+'/'+image_path+'_'+'2'+'.png'
    # fig.savefig(pa)

    # new_path = image_path+'_'+'2'+'.png'
    # finame = os.path.join(app.config['UPLOAD_FOLDER'], new_path)

    return render_template("Color.html",tables=[closestdf.to_html(classes='data')], titles=closestdf.columns.values)

@app.route("/view_dataset", methods = ['POST', 'GET'])
def view_dataset():
    return render_template("dataset.html")



if __name__ == "__main__":
    app.run(debug=True)