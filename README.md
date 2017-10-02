# seg_dna
...
## This Project works for extract DNA segments from some complex picture.

#### step 1 dig_out the DNA graph from picture and change the RGB form into a gray one

code ---- python dig_and_gray.py

results --- we have [1280, 757, 497, 569, 176, 232, 1143, 1098, 1167] parts in these 9 pics.

FIXME: more things need to do for the denoise job.

#### step 2 cut of the graph

code ---- python cut_pic.py

results --- make the pic in the center

FIXME: the pic is not nice and not in the same size.

#### step 3 find the head and get the basic statistic data

code  ---- python thinning.py

substep ---- 1) smoothy the pic; 2) thinning the graph; 3) calculate the head; 4) get the number of lines and average length

expect result ---- error is acceptable
current status ---- Something is good. Anyway some special conditions need be considered. Such as circle.
