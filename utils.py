import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2


"""
* Return all the neighbours of (x,y)
* Expect:
    -- x = the coordinate x
    -- y = the coordinate y
    -- image = the image where this point from
* Returns:
    -- [] a list concluding 8 points
"""
def neighbours(x,y,image):

    "Return 8-neighbours of image point P1(x,y), in a clockwise order"

    img = image

    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1

    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5

                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9



"""
* classify whether this point will create a cross or not
* Expect:
    -- x = the coordinate x
    -- y = the coordinate y
    --img_thin = the image after thinning
* Returns:
    -- True of False (bool value)
"""
def cross_or_not(x,y,img_thin):
    Not_sample = [[1, 0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 1, 1, 0, 0, 0],
                  [1, 1, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 1, 1, 0, 0, 1],
                  [0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0],
                  [1, 1, 0, 0, 1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 0, 0, 1],
                  [1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0],
                  [1, 1, 0, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 1],
                  [1, 0, 0, 0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0, 1, 0, 0]]  # FIXME: not complete
    if img_thin[x, y] == 1:
        if sum(utils.neighbours(x, y, img_thin)) >= 3:
            if (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[0]).all():  # FIXME: a in b
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[1]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[2]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[3]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[4]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[5]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[6]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[7]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[8]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[9]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[10]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[11]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[12]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[13]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[14]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[15]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[16]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[17]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[18]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[19]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[20]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[21]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[22]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[23]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[24]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[25]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[26]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[27]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[28]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[29]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[30]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[31]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[32]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[33]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[34]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[35]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            elif (np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1, 9)) == Not_sample[36]).all():
                # print(np.array(img_thin[x - 1:x + 2, y - 1:y + 2]).reshape((1,9)))
                # tt += 1
                return False
            else:
                return True


"""
* Incise the pic to only the dna graph
* Expect:
    -- img = img path that to be incised
* Returns:
    -- img = img after incise
"""
def incise(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_gray[img_gray >=125] = 255
    img_gray[img_gray < 125] = 0
    return img_gray[80:592,48:560]


"""
* Smooth the img 
* Expects:
    -- img_gray
* Returns:
    -- dilation img matrix
"""
def dilation_erosion(img_gray):
    kernel_2 = np.ones((2, 2), np.uint8)

    # erosion and dilation
    erosion = cv2.erode(img_gray, kernel_2, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)

    return dilation


"""
* Get the partition of img
* Expects:
    -- img_smooth
* Returns:
    -- a list of partitions [[img1],[img2],...], shape of image is (512,512)
    -- the number of partitions
"""
def partition(img_smooth):
    copy = np.copy(img_smooth)
    list_a = []
    partition_list = []
    def change(x, y):
        if copy[x, y] == 255:
            seg[x, y] = 255
            copy[x, y] = 0
            len_list = len(list_a)

            try:
                if copy[x + 1, y] == 255:
                    list_a.append([x + 1, y])
            except:
                _ = 1       # FIXME: !!!
                # print('reach the bound')
            try:
                if copy[x + 1, y + 1] == 255:
                    list_a.append([x + 1, y + 1])
            except:
                _ = 1
                # print('reach the bound')
            try:
                if copy[x, y + 1] == 255:
                    list_a.append([x, y + 1])
            except:
                _ = 1
                # print('reach the bound')
            try:
                if x > 0:
                    if copy[x - 1, y + 1] == 255:
                        list_a.append([x - 1, y + 1])
            except:
                _ = 1
                # print('reach the bound')

            try:
                if x > 0:
                    if copy[x - 1, y] == 255:
                        list_a.append([x - 1, y])
            except:
                _ = 1
                # print('reach the bound')

            try:
                if x > 0 and y > 0:
                    if copy[x - 1, y - 1] == 255:
                        list_a.append([x - 1, y - 1])
            except:
                _ = 1
                # print('reach the bound')

            try:
                if y > 0:
                    if copy[x, y - 1] == 255:
                        list_a.append([x, y - 1])
            except:
                _ = 1
                # print('reach the bound')

            try:
                if y > 0:
                    if copy[x + 1, y - 1] == 255:
                        list_a.append([x + 1, y - 1])
            except:
                _ = 1
                # print('reach the bound')
            # if len_list == len(list_a):
            return list_a[1:]
        else:
            return list_a[1:]

    while True:
        for i in range(512):
            for j in range(512):
                if copy[i, j] == 255:
                    seg = np.zeros([512, 512])
                    list_a.append([i, j])
                    while True:
                        try:
                            list_a = change(list_a[0][0], list_a[0][1])
                        except:
                            partition_list.append(seg)
                            # print('get one',len(partition_list))
                            break
        break
    return partition_list, len(partition_list)


"""
* Operation function using for thinning
"""
def transitions(neighbours):

    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"

    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2

    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


"""
* thinnning image
* Expects:
    -- image
* Returns:
    -- thinned image
"""
def thinning(image):
    image = image/255
    Image_Thinned = image.copy()  # deepcopy to protect the original image

    changing1 = changing2 = 1        #  the points to be removed (set as 0)

    while changing1 or changing2:   #  iterates until no further changes occur in the image

        # Step 1

        changing1 = []

        rows, columns = Image_Thinned.shape               # x for rows, y for columns

        for x in range(1, rows - 1):                     # No. of  rows

            for y in range(1, columns - 1):            # No. of columns

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)

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

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)

                if (Image_Thinned[x][y] == 1   and        # Condition 0

                    2 <= sum(n) <= 6  and       # Condition 1

                    transitions(n) == 1 and      # Condition 2

                    P2 * P4 * P8 == 0 and       # Condition 3

                    P2 * P6 * P8 == 0):            # Condition 4

                    changing2.append((x,y))

        for x, y in changing2:

            Image_Thinned[x][y] = 0

    return Image_Thinned


"""
* Classify the pic is a noise or not
* Expects:
    --image 
    --baseline = how many pixels are need to prove itself is a dna segment
* Returns:
    -- bool value
"""
def threshold(image, baseline):
    image = image/255
    if sum(sum(image)) < baseline:
        return False
    return True


"""
* Operate the thin_img get the stat data and coordinate of heads
* Expect:
    -- thin_img
* Return:
    -- stat = a list [number of heads, len]
    -- head_list =  a list including the coordinate of heads
"""
def head_and_len(thin_img):
    "load image data"
    Img_Original = thin_img # FIXME: change all the name
    tmp = np.zeros((Img_Original.shape[0] + 40, Img_Original.shape[1] + 40))
    tmp[20:-20, 20:-20] = Img_Original
    Img_Original = tmp

    BW_Skeleton = Img_Original

    BW_Skeleton[BW_Skeleton >= 1] = 255

    (rows, cols) = np.nonzero(BW_Skeleton)

    skel_coords = []

    for (r, c) in zip(rows, cols):

        (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1]))

        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        pix_neighbourhood = BW_Skeleton[row_neigh, col_neigh].ravel() != 0

        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r, c))

    # print("".join(["(" + str(r) + "," + str(c) + ")\n" for (r, c) in skel_coords]))

    matrix = np.zeros((512, 512))
    head_num = 0
    head_list = []
    for (r, c) in skel_coords:
        head_list.append([r,c])
    # print('this one has {} heads, {} long'.format(head_num, int(sum(sum(BW_Skeleton / 255)))))
    stat = [head_num, int(sum(sum(BW_Skeleton / 255)))]
    return stat, head_list


"""
* get the summary plot
* Expects:
    --stat_list = a list [number of heads, len]
* Returns:
    -- plot
"""
def get_summary(stat_list):
    list_line = []
    for (a,b) in stat_list:
        line_num = int(a / 2)+1
        line_avg = b/ line_num
        for i in range(line_num):
            list_line.append(line_avg)
    plt.hist(list_line)
    plt.show()
