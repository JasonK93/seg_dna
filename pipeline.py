import cv2
import skimage.io as io
import numpy as np
import logging
import os
import utils
import matplotlib.pyplot as plt

# Read img_ gray format

img_path = 'data/8.tif'
img_gray = utils.incise(img_path)


# smooth the img
img_smooth = utils.dilation_erosion(img_gray)


# Extract partitions
partitions, num_par = utils.partition(img_smooth)


# clean some partitions
clean_img = []
for i in partitions:
    if utils.threshold(i,100):
        clean_img.append(i)

# get the thinning img
thinning_image = []
for i in clean_img:
    print('thinning.......')
    thinning_image.append(utils.thinning(i))

# count the head and get stat summary
stat_list = []
for i in thinning_image:
    stat, heads = utils.head_and_len(i)
    stat_list.append(stat)

# get the summary
utils.get_summary(stat_list)


# get the single one from all


# print(len(thinning_image))
# plt.imshow(thinning_image[13])
# plt.show()