import cv2
import pandas as pd
import numpy as np

a = pd.read_csv('32_1.csv').iloc[:,1:]
b = pd.read_csv('32_2.csv').iloc[:,1:]

a = np.array(a)
b = np.array(b)

cv2.imwrite('32_1.png',a)
cv2.imwrite('32_2.png',b)