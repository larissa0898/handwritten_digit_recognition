import torch
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2 as cv2
from scipy.ndimage import center_of_mass
import matplotlib.image as mpimg


images = []
labels = []
i = 0

for digit in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    img = Image.open("initial{}.jpg".format(digit))
    img = ImageOps.invert(img)                 # invert image from white/black to black/white
    img = ImageOps.grayscale(img)              # grayscale the image

    img = ImageOps.fit(img, (20,20), Image.ANTIALIAS, bleed=0.06)    # image to 20x20, anti-alias technique and delete unnecessary black pixel at the border

    images.append(img)                        # append image to list
    labels.append(digit)                      # append digit/label to list

    img2 = ImageOps.fit(img, (28,28))            # image to 28x28
    img2.save("{}.jpg".format(digit))




""" def findCenter(img):
    #print(img.shape, img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)

img1 = Image.open("blanko_.jpg")
img1 = ImageOps.invert(img1)
img1 = ImageOps.fit(img1, (28,28))
img1 = ImageOps.grayscale(img1)
img1.save("blanko.jpg")

for m in range(len(labels)):
    img2 = mpimg.imread("initiall{}.jpg".format(labels[m]))
    cy, cx = center_of_mass(img2)


    img1 = cv2.imread("blanko.jpg")
    img2 = cv2.imread("initiall{}.jpg".format(labels[m]))

    ## (1) Find centers
    pt1 = findCenter(img1)

    ## (2) Calc offset
    dx = int(pt1[0] - cx)
    dy = int(pt1[1] - cy)

    ## (3) do slice-op `paste`
    h,w = img2.shape[:2]

    dst = img1.copy()
    dst[dy:dy+h, dx:dx+w] = img2

    cv2.imwrite("1_{}.png".format(labels[m]), dst)
    
    finalimg = Image.open("1_{}.png".format(labels[m]))
    finalimg = ImageOps.crop(finalimg, border = 0.4)
    width, height = finalimg.size
    finalimg.save("{}.png".format(labels[m]))
    print(width, height) """

    






"""     img = mpimg.imread("initiall{}.jpg".format(digit))

    cy, cx = center_of_mass(img)

    img = cv2.imread("initiall{}.jpg".format(digit))

    img1 = Image.open("blanko_.jpg")
    img1 = ImageOps.invert(img1)
    img = ImageOps.grayscale(img1)
    img1 = ImageOps.fit(img1, (28,28))
    img1.save("blanko.jpg")

    blank = mpimg.imread("blanko.jpg")

    by, bx = center_of_mass(blank)

    #img1 = Image.open("blanko_.jpg")

    dx = bx - cx
    dy = by - bx

    ## (3) do slice-op `paste`
    h,w = img.shape[:2]

    dst = blank.copy()
    dst[dy:dy+h, dx:dx+w] = img

    cv2.imwrite("1_{}.png".format(labels[m]), dst) """




    #images.append(imCrop)                        # append image to list
    #labels.append(digit)                         # append digit/label to list 
    #im = Image.fromarray(images[i])                # convert array to PIL image
    #im.save("extract{}.jpg".format(digit))          
    #i += 1