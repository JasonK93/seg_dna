import cv2
import numpy as np
import logging

logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def get_pic():
    for k in range(176):
        image = cv2.imread("pic_1/ex {}.png".format(k))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if sum(sum(gray)) > 5000:
            min_X = 255
            max_X = 0
            min_Y = 255
            max_Y = 0
            for i in range(512):
                for j in range(512):
                    if gray[i,j] == 255:
                        min_X = min(min_X, i)
                        max_X = max(max_X, i)
                        min_Y = min(min_Y, j)
                        max_Y = max(max_Y, j)

            cort_img = gray[min_X:max_X, min_Y:max_Y]
            cv2.imwrite('pic_2/{}.png'.format(k), cort_img)
            logging.info('{} pic have cut'.format(k))
if __name__ == '__main__':
    get_pic()
