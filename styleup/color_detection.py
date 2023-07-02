from sklearn.cluster import KMeans
from tensorflow.keras.utils import load_img, save_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
from math import sqrt


def open_picture(pic_name):
    # 1. Open Picture and convert it to Numpy Array
    img_path = os.path.join("..","raw_data", "test-pics",pic_name )
    img_pic = load_img(path=img_path, color_mode="rgb")
    return img_pic

def convert_to_array(img):
    return img_to_array(img).astype("int")


# Number of clusters
k_cluster = 10
def make_clusters(k_cluster, img_arr):
    clt = KMeans(n_clusters=k_cluster)
    clt_1 = clt.fit(img_arr.reshape(-1, 3))
    return clt_1


# Make a palette printing with color - also creat the list with rgb codes and percentages
def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))


    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)

    # Combine percentage and color
    dic_color_per = {}
    zipped = list(zip(list(perc.values()), k_cluster.cluster_centers_.astype("int").tolist()))
    # Sort by percentage
    list_per_color = sorted(zipped, key=lambda x: x[0], reverse=True)
    #list_per_color = [(float("%.2f" % x[0]), x[1]) for x in list_per_color]

    return palette, list_per_color

# Run it
#palette, dic_color_per = palette_perc(clt_1)


# Put together dictionary with RBG codes to color names in string
def get_color_names():
    """
    Creates a dictionary with colar names and RGB codes from and csv file
    """

    # Path to file with RGB codes and color names
    color_map_path = os.path.join("..","raw_data", "color","color-map.csv" )
    # Make a dataframe
    df = pd.read_csv(color_map_path,sep=";")
    df['rgb-code'] = df['rgb-code'].apply(lambda x: x.strip("''"))

    # Create Dictionary with color names and rgb codes
    color_dict = {}
    for index, row in df.iterrows():
        color_dict[df.loc[index, 'c-name']] = [int(x) for x in df.loc[index, 'rgb-code'].replace("(","").replace(")","").split(",")]

    return color_dict

# run it
#color_dict = get_color_names()


# Find the closest color for an rgb code
def closest_color(rgb,color_dict):
    """
    Finds the closest color for an RGB code in the color dictionary
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    color_diffs = []

    for key, value in color_dict.items():
        cr = color_dict[key][0]
        cg = color_dict[key][1]
        cb = color_dict[key][2]
        color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, key))
    return min(color_diffs)[1]


# Get the color names for our complete results
def all_color_names(dic_color_per,color_dict):
    """
    Gets all colornames as String for a list with RGB colors
    """
    res_list = []
    for res_color in dic_color_per:
        color_name = closest_color(res_color[1],color_dict)
        res_list.append([color_name, res_color[0], res_color[1]])

    return res_list

# Run it
#pic_cname_perc_rgb = all_color_names(dic_color_per,color_dict)


# ------------------------------------------ #
# 2. complementary colors

# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

# This function takes a [r,g,b] list and returns the complement color as list
def complement(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    k = hilo(r, g, b)
    return list(k - u for u in (r, g, b))

# Creat list with complementary colors
def list_complentary_colors(dic_color_per,color_dict):
    res_list_cc = []
    for color in dic_color_per:
        c_rgb = complement(color[1])
        perc = color[0]
        c_name = closest_color(c_rgb,color_dict)

        res_list_cc.append([c_name, perc, c_rgb])
    return res_list_cc

# run it
#comp_colors_cname_perc_rgb = list_complentary_colors(dic_color_per,color_dict)


# 3. Make the final list with colors

# Filter out some colors - wall color and black, grey, white
def color_below_percentage(color_list):

    res_list = []
    for color in color_list:
        if "black" in color[0].lower() or "grey" in color[0].lower() \
        or "white" in color[0].lower() or "gray" in color[0].lower() \
        or "silver" in color[0].lower() or "brown" in color[0].lower():
            pass
        #elif color[1] > 0.15:
        #    pass
        else:
            res_list.append(color)
    return res_list




def final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb,colors_real_pic,colors_complementary):

    # Select the number of colors to use
    i = 0
    c_i = 0

    res_list = []

    # Add two from the real picture colors
    for color in color_below_percentage(pic_cname_perc_rgb):
        if i < colors_real_pic:
            res_list.append(color)
            i += 1

    # Add one picture from the complementary colors
    for color_c in color_below_percentage(comp_colors_cname_perc_rgb):
        if c_i < colors_complementary:
            res_list.append(color_c)
            c_i += 1


    return res_list[:3]

# run it
#final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb)

# Just return color names of final result
def list_final_color_names(final_list_colors_with_rgb):
    res_list = []
    for color in final_list_colors_with_rgb:
        res_list.append(color[0])
    return res_list






if __name__ == '__main__':
    try:
        palette, dic_color_per = palette_perc(clt_1)
        color_dict = get_color_names()
        pic_cname_perc_rgb = all_color_names(dic_color_per,color_dict)
        comp_colors_cname_perc_rgb = list_complentary_colors(dic_color_per,color_dict)
        print("final result")
        final_color_list = final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb)

    except:
        pass
