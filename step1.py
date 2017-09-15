import cv2
import numpy as np

img_rgb = cv2.imread('3.tif')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
kernel_2 = np.ones((1,1),np.uint8)



HSV = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)



# gradX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
# gradient = cv2.subtract(gradX, gradY)
# gradient = cv2.convertScaleAbs(gradient)
# #
Lower = np.array([22,20,20])
Upper = np.array([30,255,255])

mask = cv2.inRange(HSV,Lower, Upper)

erosion = cv2.erode(mask, kernel_2, iterations=1)
erosion = cv2.erode(erosion, kernel_2, iterations=1)
dilation = cv2.dilate(erosion, kernel_2, iterations=1)
dilation = cv2.dilate(dilation, kernel_2, iterations=1)

target = cv2.bitwise_and(img_rgb, img_rgb, mask=dilation)



img_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
gradX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (2, 2))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=1)
closed = cv2.dilate(closed, None, iterations=1)

# cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#
# # compute the rotated bounding box of the largest contour
# rect = cv2.minAreaRect(c)
# box = np.int0(cv2.cv.BoxPoints(rect))
#
# # draw a bounding box arounded the detected barcode and display the image
# cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
# cv2.imwrite("contoursImage2.jpg", image)
# cv2.waitKey(0)

#

cv2.imwrite('test.png',closed)
