import torch
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2 as cv2
from scipy.ndimage import center_of_mass
import matplotlib.image as mpimg


###########################################################
# zweite Variante des Pre-Processings
###########################################################

def secondpreprocessing(image):
    img = ImageOps.invert(image)               # invert image from white/black to black/white
    img = ImageOps.grayscale(img)              # grayscale the image

    img = ImageOps.fit(img, (20,20), Image.ANTIALIAS, bleed=0.06)    # image to 20x20, anti-alias technique and delete unnecessary black pixel at the border

    img2 = ImageOps.fit(img, (28,28))            # image to 28x28

    return img2



###########################################################
# erste Variante des Pre-Processings
###########################################################

def findCenter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)


def firstpreprocessing(image):
    img = ImageOps.invert(image)
    img.save("initiall.jpg")

    img = cv2.imread("initiall.jpg")       # load image

    blur = cv2.GaussianBlur(img,(5,5),0)      # apply Gaussian Blur

    extract = cv2.selectROI(blur)                         # extract digit with ROI
    imCrop = blur[int(extract[1]):int(extract[1]+extract[3]), int(extract[0]):int(extract[0]+extract[2])]      # save it in variable imCrop
 
    im = Image.fromarray(imCrop)          # convert array to PIL image
    im.save("extract.jpg")          


    # find the center of digit, extract and put it in center of new square image

    img1 = Image.open("blanko_.jpg")
    img1 = ImageOps.invert(img1)
    img1 = ImageOps.fit(img1, (28,28))
    img1.save("blanko.jpg")

    img1 = cv2.imread("blanko.jpg")
    img2 = cv2.imread("extract.jpg")
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

    cv2.imwrite("laststep.png".format(labels[m]), dst)


    # resize image to 28x28

    image = Image.open("laststep.png".format(labels[img])).convert('L')          # open image and grayscale it

    resized = image.resize((28,28))

    return resized