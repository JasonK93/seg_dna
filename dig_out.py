import cv2
import numpy as np

def incise():
    for i in range(1,4):
        img_rgb = cv2.imread('{}.tif'.format(i))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('square_{}.png'.format(i),img_gray[80:592,48:560])


def process():
    kernel_2 = np.ones((2, 2), np.uint8)

    # high means bright
    img_rgb = cv2.imread('square_3.png')
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
    cv2.imwrite('test.png', dilation)
    return dilation

def extract(dilation):
    num = 0
    copy = np.copy(dilation)
    def change(x,y):
        seg[x,y] = 255
        copy[x,y] = 0
        try:
            if copy[x+1,y] == 255:
                change(x+1,y)
        except:
            print('reach the bound')
        try:
            if copy[x+1,y+1] == 255:
                change(x+1, y+1)
        except:
            print('reach the bound')
        try:
            if copy[x,y+1] == 255:
                change(x, y+1)
        except:
            print('reach the bound')
        try:
            if x > 0:
                if copy[x-1, y+1] == 255:
                    change(x-1,y+1)
        except:
            print('reach the bound')

        try:
            if x > 0:
                if copy[x-1,y] == 255:
                    change(x-1,y)
        except:
            print('reach the bound')

        try:
            if x >0 and y >0:
                if copy[x-1,y-1] == 255:
                    change(x-1,y-1)
        except:
            print('reach the bound')

        try:
            if y >0 :
                if copy[x,y-1] == 255:
                    change(x, y-1)
        except:
            print('reach the bound')


        try:
            if y>0 :
                if copy[x+1,y-1] == 255:
                    change(x+1, y-1)
        except:
            print('reach the bound')


    while True:
        # seg = np.zeros([512,512])
        for i in range(512):
            # seg = np.zeros([512, 512])
            for j in range(512):
                if copy[i,j] == 255:
                    seg = np.zeros([512, 512])
                    change(i,j)
                    cv2.imwrite('ex {}.png'.format(num), seg)
                    num += 1

        break





if __name__ == '__main__':
    # incise()
    dilation = process()
    extract(dilation)

