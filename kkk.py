import cv2
import numpy as np
import logging

logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

image = cv2.imread("pic_1/ex 4.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray[0] = [0]*512

kernel_2 = np.ones((3,3),np.uint8)


# erosion = cv2.erode(gray, kernel_2, iterations=1)
# erosion = cv2.erode(erosion, kernel_2, iterations=1)
dilation = cv2.dilate(gray, kernel_2, iterations=1)
# dilation = cv2.dilate(dilation, kernel_2, iterations=1)
erosion = cv2.erode(dilation, kernel_2, iterations=1)
dilation = cv2.dilate(erosion, kernel_2, iterations=10)

cv2.imwrite('tttttt.png', dilation)

# thumbnail = cv2.CreateMat(im.rows/10, im.cols/10, cv.CV_8UC3)
# cv2.Resize(im, thumbnail)