import random
import math
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

# return (merged image, move, deg, rate)
def merge(im1, im2, match):
    if len(match) == 0:
        merged = Image.new('RGB', (im1.width+im2.width, max(im1.height, im2.height)))
        merged.paste(im1, (0, 0))
        merged.paste(im2, (im1.width, 0))
        return (merged, (im1.width, 0), 0, 1)
    if len(match) == 1:
        dx = match[0].data1.x - match[0].data2.x
        dy = match[0].data1.y - match[0].data2.y
        mgnw = im2.width//2
        mgnh = im2.height//2
        merged = Image.new('RGB', (im1.width+im2.width, im1.height+im2.height))
        merged.paste(im2, (dy+mgnw, dx+mgnh))
        merged.paste(im1, (mgnw, mgnh))
        return (merged, (dy, dx), 0, 1)


    # find bestmatching
    bestmatch = 0
    secondmatch = 0
    diff_b = 1000000
    diff_s = 1000000
    for i in range(len(match)):
        m = match[i]

        im1_arr = np.array(im1)[m.data1.x-2:m.data1.x+3, m.data1.y-2:m.data1.y+3]
        im2_arr = np.array(im2)[m.data2.x-2:m.data2.x+3, m.data2.y-2:m.data2.y+3]

        d = ((im1_arr.mean(0).mean(0) - im2_arr.mean(0).mean(0))**2).sum()

        if d < diff_b:
            secondmatch = bestmatch
            bestmatch = i
            diff_s = diff_b
            diff_b = d
        elif d < diff_s:
            secondmatch = i
            diff_s = d

    b1 = match[bestmatch].data1
    s1 = match[secondmatch].data1
    b2 = match[bestmatch].data2
    s2 = match[secondmatch].data2

    # euclid distance
    d = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    # calc rate
    r = 1.5
    if d(b2, s2) != 0:
        r = d(b1, s1) / d(b2, s2) 
        r = int(r*10 + 0.5) / 10

    # calc degree
    theta = math.atan2(b2.x-s2.x, b2.y-s2.y) - math.atan2(b1.x-s1.x, b1.y-s1.y)
    deg = ((theta*180)/math.pi) % 360
    deg = int(deg*10 + 0.5) / 10

    # transform im2
    new_im = im2.rotate(deg, center=(b2.y, b2.x), expand=True).resize((int(im2.width*r+0.5), int(im2.height*r+0.5)))
    new_bestx = int(b2.x*r + 0.5)
    new_besty = int(b2.y*r + 0.5)

    # merge
    dy = b1.y-new_besty
    dx = b1.x-new_bestx
    mgnw = new_im.width//2
    mgnh = new_im.height//2
    merged = Image.new('RGB', (im1.width+new_im.width, im1.height+new_im.height))
    merged.paste(new_im, (dy+mgnw, dx+mgnh))
    merged.paste(im1, (mgnw, mgnh))

    return (merged, (dy, dx), deg, r)
