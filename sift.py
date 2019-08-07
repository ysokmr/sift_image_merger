from PIL import Image, ImageFilter, ImageChops
import numpy as np
import math
import cv2

import matplotlib.pyplot as plt

class descriptor:
    def __init__(self, x, y, des):
        self.x = x
        self.y = y
        self.des = des


class SIFT:
    def __init__(self, im, sigma=1.6, k=math.sqrt(2), scales=8, blksize=8):
        self.im = im
        self.sigma = 1.6
        self.k = k
        self.scales = scales
        self.blksize = blksize
        self.dog = []
        self.keyPoints = []
        self.feature = []
        self.descriptor = []


    def harrisEdge(self, i, j, k):
        f = lambda x, y: int(self.dog[i][x][y])
        # differential
        dx = f(j+1, k) - f(j, k)
        dxx = f(j+1, k) + f(j-1, k) - 2*f(j, k)
        dy = f(j, k+1) - f(j, k)
        dyy = f(j, k+1) + f(j, k-1) - 2*f(j, k)

        h = np.array([[dxx, dx*dy], [dx*dy, dyy]])
        det = np.linalg.det(h)
        if det == 0: return False
        r = np.trace(h) / det
        gamma = 10 # eigenvalue rate
        return r >= ((gamma + 1)**2)/gamma

    # generate DoG images
    def makeDoG(self):
        self.dog = []
        g1 = self.im.convert("L").filter(ImageFilter.GaussianBlur(self.sigma))
        for i in range(1, self.scales+1):
            g2 = self.im.convert("L").filter(ImageFilter.GaussianBlur(self.sigma*(self.k**(i))))
            self.dog.append(np.array(g2) - np.array(g1))
            g1 = g2
        self.dog = np.array(self.dog)

        # for d in self.dog:
        #     plt.imshow(d)
        #     plt.show()


    def findKeyPoints(self):
        def isKeyPoint(i, j, k):
            t = 5 # threshold
            p = self.dog[i][j][k]
            if p < t or p > 255-t: return False
            ismax = (np.sum(p > self.dog[i-1:i+2, j-1:j+2, k-1:k+2]) == 26)
            ismin = (np.sum(p < self.dog[i-1:i+2, j-1:j+2, k-1:k+2]) == 26)
            return ismax or ismin

        self.keyPoints = []
        w = self.blksize*2
        for i in range(1, len(self.dog)-1):
            for j in range(w, self.im.height-w):
                for k in range(w, self.im.width-w):
                    if isKeyPoint(i, j, k):
                        self.keyPoints.append((i, j, k))

    def calcOrientation(self, im, j, k):
        l = lambda x, y: int(im[x][y])
        # g = math.exp(-(j**2 + k**2)/(2*self.sigma*(self.k**i))) / math.sqrt(2*math.pi*self.sigma*(math.sqrt(2)**i))

        m = math.sqrt((l(j+1, k)-l(j-1, k))**2 + (l(j, k+1)-l(j, k-1))**2)
        theta = math.atan2(l(j, k+1)-l(j, k-1), l(j+1, k)-l(j-1, k))
        # w = g*m
        return (m, theta)

    def calcHistgram(self, im, j, k, orient_num):
        w = self.blksize
        # hist[i]: weight of i-5 <= rad < i+5
        hist = np.zeros(orient_num)
        x_min = max(1, j-w//2)
        x_max = min(len(im)-2, j+w//2)
        y_min = max(1, k-w//2)
        y_max = min(len(im[j])-2, k+w//2)
        r = (2*math.pi)/orient_num
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                w, theta = self.calcOrientation(im, x, y)
                o = (theta + r/2) % (2*math.pi)
                hist[int(o//r)] += w
        return hist


    def findOrientPeek(self, i, j, k, partition=36):
        hist = self.calcHistgram(self.dog[i], j, k, partition)
        # find max spectrum
        max_i = 0
        w = 0
        for l in range(len(hist)):
            if hist[l] > w:
                w = hist[l]
                max_i = l
        theta = max_i * ((2*math.pi)/partition)
        return (w, theta)

    def calcDescriptor(self, i, j, k, o):
        # rotate image
        im = Image.fromarray(self.dog[i], "L").rotate(o, expand=True, center=(j, k))
        im = np.array(im)

        w = self.blksize
        des = []
        for x in range(-w*2, w*2, w):
            for y in range(-w*2, w*2, w):
                hist = self.calcHistgram(im, j+x, k+y, 8)
                des.append(hist)
        return np.array(des)

    # return (keyPoints, feature, descriptor)
    def apply(self):
        self.makeDoG()
        self.findKeyPoints()
        self.feature = []
        self.descriptor = []
        i = 0
        while i < len(self.keyPoints):
            p = self.keyPoints[i]
            # remove edge
            if self.harrisEdge(p[0], p[1], p[2]):
                del self.keyPoints[i]
                continue
            w, theta = self.findOrientPeek(p[0], p[1], p[2])
            # remove point where weight=0
            #if w == 0:
            #    del self.keyPoints[i]
            #    continue
            self.feature.append((w, theta))
            des = self.calcDescriptor(p[0], p[1], p[2], theta)
            # remove point where descriptor=0
            if (des == 0).all():
                del self.keyPoints[i]
                del self.feature[i]
                continue
            self.descriptor.append(descriptor(p[1], p[2], des))
            i += 1
        return (self.keyPoints, self.feature, self.descriptor)
