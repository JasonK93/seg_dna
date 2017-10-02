import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2


def neighbours(x,y,image):

    "Return 8-neighbours of image point P1(x,y), in a clockwise order"

    img = image

    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1

    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5

                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9
