import cv2
import numpy as np
import logging

logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

image = cv2.imread("tttttt.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
copy = np.copy(gray)
shape_x, shape_y = gray.shape[0], gray.shape[1]

list_a = []


def change(x, y, head):
    if copy[x, y] == 255:
        copy[x, y] = 0
        len_list = len(list_a)
        try:
            if copy[x + 1, y] == 255:
                list_a.append([x + 1, y])
        except:
            print('reach the bound')
        try:
            if copy[x + 1, y + 1] == 255:
                list_a.append([x + 1, y + 1])
        except:
            print('reach the bound')
        try:
            if copy[x, y + 1] == 255:
                list_a.append([x, y + 1])
        except:
            print('reach the bound')
        try:
            if x > 0:
                if copy[x - 1, y + 1] == 255:
                    list_a.append([x - 1, y + 1])
        except:
            print('reach the bound')

        try:
            if x > 0:
                if copy[x - 1, y] == 255:
                    list_a.append([x - 1, y])
        except:
            print('reach the bound')

        try:
            if x > 0 and y > 0:
                if copy[x - 1, y - 1] == 255:
                    list_a.append([x - 1, y - 1])
        except:
            print('reach the bound')

        try:
            if y > 0:
                if copy[x, y - 1] == 255:
                    list_a.append([x, y - 1])
        except:
            print('reach the bound')

        try:
            if y > 0:
                if copy[x + 1, y - 1] == 255:
                    list_a.append([x + 1, y - 1])
        except:
            print('reach the bound')
        if len_list == len(list_a):
            head += 1
        return list_a[1:], head
    else:
        return list_a[1:], head

head = 0
for i in range(shape_x):
    for j in range(shape_y):
        if copy[i,j] == 255:
            list_a.append([i,j])
            while True:
                try:
                    list_a,head = change(list_a[0][0], list_a[0][1],head)
                except:
                    break
print('the num of heads is {}'.format(head))