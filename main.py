from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import math
import sift
import knn
import merge
import matplotlib.pyplot as plt

import random
import time

lv = input('Level: ')
num = input('num: ')

target = './sample_img/Level{0}/{0}-{1}'.format(lv, num)
if lv == 'my':
    target = './myimage/{}'.format(num)
    im1 = Image.open(target + '-1.jpg')
    im2 = Image.open(target + '-2.jpg')
else:
    im1 = Image.open(target + '-1.ppm')
    im2 = Image.open(target + '-2.ppm')

t1 = time.perf_counter()

# SIFT
keyPoints1, features1, des1 = sift.SIFT(im1).apply()
keyPoints2, features2, des2 = sift.SIFT(im2).apply()

t2 = time.perf_counter()
print("SIFT time: " + str(t2 - t1))

# feature point matching
pair = knn.KNN(des1, des2, lambda d1, d2: np.power(d1.des-d2.des, 2).sum()).apply(2)
good = []
for best, n in pair:
    if best.dist < 0.7*n.dist:
        good.append(best)

# merge
result = merge.merge(im1, im2, good)
t3 = time.perf_counter()
res_txt = 'X-Offset : {}\n'.format(result[1][0])
res_txt += 'Y-Offset : {}\n'.format(result[1][1])
res_txt += 'Angle : {}\n'.format(result[2])
res_txt += 'Ratio : {}\n'.format(result[3])
res_txt += 'SIFT time : {}\n'.format(t2-t1)
res_txt += 'total time : {}'.format(t3-t1)
print('result: ({0}, {1}, {2}, {3})'.format(result[1][0], result[1][1], result[2], result[3]))
print('total time: ' + str(t3 - t1))

result_file = './result_img/Level{0}/{0}-{1}'.format(lv, num)
if lv == 'my':
    result_file = './myimage/{}'.format(num)
    result[0].save(result_file + '.jpg')
else:
    result[0].save(result_file + '.ppm')
with open(result_file + '.txt', 'w') as f:
    f.write(res_txt)

plt.imshow(result[0])
plt.show()



# draw matches
bestmatch = 0
secondmatch = 0
diff_b = 1000000
diff_s = 1000000
for i in range(len(good)):
    m = good[i]

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

best = good[bestmatch]
second = good[secondmatch]

bx1, by1 = (best.data1.x, best.data1.y)
bx2, by2 = (best.data2.x, best.data2.y)
print("bestmatch diff: ({0}, {1})".format(by1 - by2, bx1 - bx2))

sx1, sy1 = (second.data1.x, second.data1.y)
sx2, sy2 = (second.data2.x, second.data2.y)
print("secondmatch diff: ({0}, {1})".format(sy1 - sy2, sx1 - sx2))

ans = {}
ans_txt = ''
with open(target + '.txt', 'r') as f:
    ans_txt = f.read().split('\n')
for a in ans_txt:
    if not ':' in a: continue
    tmp = a.split(':')
    ans[tmp[0].strip()] = float(tmp[1].strip())

print("answer: ({0}, {1}, {2}, {3})".format(ans['X-Offset'], ans['Y-Offset'], ans['Angle'], ans['Ratio']))



draw1 = ImageDraw.Draw(im1)
draw2 = ImageDraw.Draw(im2)
# draw keypoint
for i in range(len(keyPoints1)):
    s, x, y = keyPoints1[i]
    w, theta = features1[i]
    draw1.ellipse((y-1, x-1, y+1, x+1), fill=(255, 255, 255))
    # draw1.line((y, x, y+int(10*math.sin(theta)), x+int(10*math.cos(theta))), fill=(255, 0, 0), width=1)

for i in range(len(keyPoints2)):
    s, x, y = keyPoints2[i]
    w, theta = features2[i]
    draw2.ellipse((y-1, x-1, y+1, x+1), fill=(255, 255, 255))
    # draw2.line((y, x, y+int(10*math.sin(theta)), x+int(10*math.cos(theta))), fill=(255, 0, 0), width=1)

# draw matching point
for i in range(len(good)):
    x = good[i].data1.x
    y = good[i].data1.y
    draw1.ellipse((y-1, x-1, y+1, x+1), fill=(255, 255*i//len(good), 255 - 255*i//len(good)))
    x = good[i].data2.x
    y = good[i].data2.y
    draw2.ellipse((y-1, x-1, y+1, x+1), fill=(255, 255*i//len(good), 255 - 255*i//len(good)))

# draw bestmatch
draw1.ellipse((by1-1, bx1-1, by1+1, bx1+1), fill=(0, 255, 255))
draw2.ellipse((by2-1, bx2-1, by2+1, bx2+1), fill=(0, 255, 255))

# draw secondmatch
draw1.ellipse((sy1-1, sx1-1, sy1+1, sx1+1), fill=(0, 255, 255))
draw2.ellipse((sy2-1, sx2-1, sy2+1, sx2+1), fill=(0, 255, 255))

res = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
res.paste(im1, (0, 0))
res.paste(im2, (im1.width, 0))
draw_res = ImageDraw.Draw(res)
for g in good:
    draw_res.line((g.data1.y, g.data1.x, g.data2.y+im1.width, g.data2.x), fill=(255, 0, 0), width=3)
draw_res.line((by1, bx1, by2+im1.width, bx2), fill=(0, 255, 255), width=3)
draw_res.line((sy1, sx1, sy2+im1.width, sx2), fill=(0, 255, 255), width=3)

plt.imshow(res)
plt.show()
