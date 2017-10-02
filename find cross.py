import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
import utils

Not_sample = [[1, 0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1],
              [0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 1, 0],
              [0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 0, 0, 1],
              [1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 1, 0],
              [0, 0, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 1, 0, 0]] # FIXME: not complete
Not_sample = np.array(Not_sample)

def transitions(neighbours):

    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"

    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2

    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangSuen(image):

    "the Zhang-Suen Thinning Algorithm"

    Image_Thinned = image.copy()  # deepcopy to protect the original image

    changing1 = changing2 = 1        #  the points to be removed (set as 0)

    while changing1 or changing2:   #  iterates until no further changes occur in the image

        # Step 1

        changing1 = []

        rows, columns = Image_Thinned.shape               # x for rows, y for columns

        for x in range(1, rows - 1):                     # No. of  rows

            for y in range(1, columns - 1):            # No. of columns

                P2,P3,P4,P5,P6,P7,P8,P9 = n = utils.neighbours(x, y, Image_Thinned)

                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions

                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6

                    transitions(n) == 1 and    # Condition 2: S(P1)=1

                    P2 * P4 * P6 == 0  and    # Condition 3

                    P4 * P6 * P8 == 0):         # Condition 4

                    changing1.append((x,y))

        for x, y in changing1:

            Image_Thinned[x][y] = 0

        # Step 2

        changing2 = []

        for x in range(1, rows - 1):

            for y in range(1, columns - 1):

                P2,P3,P4,P5,P6,P7,P8,P9 = n = utils.neighbours(x, y, Image_Thinned)

                if (Image_Thinned[x][y] == 1   and        # Condition 0

                    2 <= sum(n) <= 6  and       # Condition 1

                    transitions(n) == 1 and      # Condition 2

                    P2 * P4 * P8 == 0 and       # Condition 3

                    P2 * P6 * P8 == 0):            # Condition 4

                    changing2.append((x,y))

        for x, y in changing2:

            Image_Thinned[x][y] = 0

    return Image_Thinned


def test():
    img = io.imread('ex 38.png')
    img = img/255

    img_thin = zhangSuen(img)
    # img_thin = cv2.resize(img_thin, (256,256))
    # img_thin = zhangSuen(img_thin)
    # img_thin[img_thin >= 0.2] = 1
    # img_thin[img_thin < 0.2] = 0
    # img_thin = zhangSuen(img_thin)
    cross_list = []
    tmp = 0
    tt = 0
    num_pic = 0
    for x in range(512):
        for y in range(512):
            if img_thin[x,y] == 1:
                if sum(utils.neighbours(x,y,img_thin)) >=3:
                    if (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[0]).all():  # FIXME: a in b
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[1]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[2]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[3]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[4]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[5]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[6]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[7]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[8]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[9]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[10]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[11]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[12]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[13]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[14]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[15]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[16]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[17]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[18]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[19]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[20]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[21]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[22]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[23]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[24]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[25]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[26]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[27]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[28]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[29]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[30]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)) == Not_sample[31]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt+=1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[32]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt += 1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[33]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt += 1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[34]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt += 1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[35]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt += 1
                    elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[36]).all():
                        # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                        tt += 1
                    else:
                        tmp = tmp + img_thin[x, y]
                        cross_list.append([x,y])
                        print(img_thin[x - 1:x + 2,
                              y - 1:y + 2])
                        tm = np.copy(img_thin)
                        cv2.rectangle(tm, (y-5, x-5), (y+6, x+6), 0.5, 1)
                        tm[tm > 0] = 255
                        cv2.imwrite('bound/{}.png'.format(num_pic),tm)
                        num_pic +=1
                        # cv2.rectangle(img_thin, (y-5, x-5), (y+6, x+6), 0.1, 1)
                        # plt.imshow(tm)
                        # plt.show()
    print(len(cross_list))
    print('tmp', tmp)
    print('tt',tt)
    # img_thin = zhangSuen(img_thin)

    # plt.imshow(img_thin)
    # plt.show()

if __name__ == '__main__':
    test()
