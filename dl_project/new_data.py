import torch
from PIL import Image
from PIL import ImageShow
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import cv2 as cv2
import math


""" images = []
labels = []
i = 0

for no in [1, 4, 8, 9]:
    img = cv2.imread(str(no)+".jpg")

    blur = cv2.GaussianBlur(img,(5,5),0)
    #extract = cv2.selectROI(blur)
    images.append(blur)
    labels.append(no)
    i += 1

#print(images[0])
#print(labels) """

""" images = []
labels = []
i = 0
#1, 4, 8, 9
for no in [4]:
    img = cv2.imread(str(no)+".jpg")

    blur = cv2.GaussianBlur(img,(5,5),0)
    extract = cv2.selectROI(blur)
    imCrop = blur[int(extract[1]):int(extract[1]+extract[3]), int(extract[0]):int(extract[0]+extract[2])]
    images.append(imCrop)
    labels.append(no)
    i += 1

#print(type(images[0]))
#print(images[0])
#print(labels)

im = Image.fromarray(images[0])
im.save("extract4.jpeg") """

""" 
pyplot.imshow(images[0])
pyplot.show()
""" 
""" def findCenter(img):
    print(img.shape, img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    #cv2.imshow("threshed", threshed);cv2.waitKey();cv2.destroyAllWindows()
    #_, cnts, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)

img1 = cv2.imread("blanko_.jpg")
img2 = cv2.resize(cv2.imread("extract4.jpeg"), None, fx=0.3, fy=0.3)

## (1) Find centers
pt1 = findCenter(img1)
pt2 = findCenter(img2)

## (2) Calc offset
dx = pt1[0] - pt2[0]
dy = pt1[1] - pt2[1]

## (3) do slice-op `paste`
h,w = img2.shape[:2]

dst = img1.copy()
dst[dy:dy+h, dx:dx+w] = img2

cv2.imwrite("vier.png", dst) """

""" 
pyplot.imshow()
pyplot.show() """

image = Image.open("vier.png")

resized = image.resize((28,28))

pix = np.array(resized)
#print(type(pix))
#print(pix)
pyplot.imshow(pix)
pyplot.show()