import cv2
import numpy as np
import logging
import os

logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def incise():
    os.mkdir('data/square')
    for i in range(1,9):
        img_rgb = cv2.imread('data\{}.tif'.format(i))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('data/square/square_{}.png'.format(i),img_gray[80:592,48:560])


def process():
    os.mkdir('data/gray')
    dilation_list = []
    for i in range(1,9):
        kernel_2 = np.ones((2, 2), np.uint8)

        # high means bright
        img_rgb = cv2.imread('square_{}.png'.format(i))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        HSV = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        Lower = np.array([0, 0, 125])
        Upper = np.array([255, 255, 255])

        mask = cv2.inRange(HSV, Lower, Upper)

        # erosion and dilation
        # erosion = cv2.erode(mask, kernel_2, iterations=1)
        # erosion = cv2.erode(erosion, kernel_2, iterations=1)
        # dilation = cv2.dilate(erosion, kernel_2, iterations=1)
        # dilation = cv2.dilate(dilation, kernel_2, iterations=1)

        dilation = cv2.dilate(mask, kernel_2, iterations=1)
        cv2.imwrite('data/gray/gray{}.png'.format(i), dilation)
        dilation_list.append(dilation)
    return dilation_list


def extract(dilation):
    num_list = []
    os.mkdir('data/step1')
    for pic_num in range(1,9):
        os.mkdir('data/step1/pic{}'.format(pic_num))
        num = 0
        copy = np.copy(dilation[pic_num])
        list_a = []
        def change(x,y):
            if copy[x,y] == 255:
                seg[x,y] = 255
                copy[x,y] = 0
                len_list = len(list_a)

                try:
                    if copy[x+1, y] == 255:
                        list_a.append([x+1, y])
                except:
                    print('reach the bound')
                try:
                    if copy[x+1,y+1] == 255:
                        list_a.append([x + 1, y+1])
                except:
                    print('reach the bound')
                try:
                    if copy[x,y+1] == 255:
                        list_a.append([x, y + 1])
                except:
                    print('reach the bound')
                try:
                    if x > 0:
                        if copy[x-1, y+1] == 255:
                            list_a.append([x - 1, y + 1])
                except:
                    print('reach the bound')

                try:
                    if x > 0:
                        if copy[x-1,y] == 255:
                            list_a.append([x - 1, y])
                except:
                    print('reach the bound')

                try:
                    if x >0 and y >0:
                        if copy[x-1,y-1] == 255:
                            list_a.append([x - 1, y - 1])
                except:
                    print('reach the bound')

                try:
                    if y >0 :
                        if copy[x,y-1] == 255:
                            list_a.append([x, y - 1])
                except:
                    print('reach the bound')


                try:
                    if y > 0:
                        if copy[x+1,y-1] == 255:
                            list_a.append([x + 1, y - 1])
                except:
                    print('reach the bound')
                # if len_list == len(list_a):
                return list_a[1:]
            else:
                return list_a[1:]

        while True:
            # seg = np.zeros([512,512])
            for i in range(512):
                # seg = np.zeros([512, 512])
                for j in range(512):
                    if copy[i,j] == 255:
                        seg = np.zeros([512, 512])
                        list_a.append([i, j])
                        while True:
                            try:
                                list_a = change(list_a[0][0], list_a[0][1])

                            except:
                                logging.info('get {} pic'.format(num+1))
                                cv2.imwrite('data/step1/pic_1/ex {}.png'.format(num), seg)

                                num += 1
                                break

            num_list.append(num)
        return num_list



if __name__ == '__main__':
    if not os.path.exists('data\square'):
        incise()
    if not os.path.exists('data\gray'):
        dilation = process()
        num = extract(dilation)
        print(num)
