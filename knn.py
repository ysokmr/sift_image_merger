import numpy as np
import math


class Match:
    def __init__(self, data1, data2, dist):
        self.data1 = data1
        self.data2 = data2
        self.dist = dist


class KNN:
    def __init__(self, datas1, datas2, distf=lambda x, y: math.sqrt(x**2 + y**2)):
        self.data1 = datas1
        self.data2 = datas2
        self.distf = distf

    def apply(self, k=3):
        matches = []
        for d1 in self.data1:
            sort = []
            for d2 in self.data2:
                dist = self.distf(d1, d2)
                sort.append(Match(d1, d2, dist))
            sort.sort(key=lambda x: x.dist)
            matches.append(sort)
        return list(map(lambda x: x[:k], matches))

