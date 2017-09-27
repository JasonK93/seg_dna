import cv2
import numpy as np
import logging
import os
logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def get_pic():
    os.mkdir('data/step2')
    for pic_num in range(9):
        os.mkdir('data/step2/pic{}'.format(pic_num))
        for k in range(1500):
            try:
                image = cv2.imread("data/step1/pic{}/ex {}.png".format(pic_num, k))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if sum(sum(gray)) > 5000:    # FIXME: different condition need different threshold
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
                    # tmp = np.zeros((512,512))
                    # tmp[256-(min_X+max_X)/2:256+(min_X+max_X)/2, 256-(min_Y+max_Y)/2:256+(min_Y+max_Y)/2] = cort_img
                    cv2.imwrite('data/step2/pic{}/{}.png'.format(pic_num, k), cort_img)
                    logging.info('{} pic have cut'.format(k))
            except:
                break
if __name__ == '__main__':
    get_pic()
