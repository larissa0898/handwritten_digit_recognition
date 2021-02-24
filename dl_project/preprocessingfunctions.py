from PIL import Image
from PIL import ImageOps
import cv2 as cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import center_of_mass
import matplotlib.image as mpimg


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
    mg = ~image
    mg = cv2.GaussianBlur(mg,(19,19),0)

    img1 = cv2.imread("blanko.jpg")
    img1 = ~img1                  # invert image

    img2 = cv2.resize(mg, None, fx=0.77, fy=0.77)
    cv2.imwrite("initiall.jpg", img2)

    img2 = mpimg.imread("initiall.jpg")

    ## (1) Find centers
    pt1 = findCenter(img1)
    pt2 = ndi.center_of_mass(img2)

    img2 = cv2.imread("initiall.jpg")
    ## (2) Calc offset
    dx = pt1[0] - int(pt2[0])
    dy = pt1[1] - int(pt2[1])

    ## (3) do slice-op `paste`
    h,w = img2.shape[:2]

    dst = img1.copy()
    dst[dy:dy+h, dx:dx+w] = img2

    cv2.imwrite("laststep.jpg", dst)


    # resize image to 28x28

    image = Image.open("laststep.jpg").convert('L')         # open image and grayscale it

    resized = image.resize((28,28))

    return resized



###########################################################
# zweite Variante des Pre-Processings
###########################################################

def secondpreprocessing(image):
    img = Image.fromarray(image)
    img = ImageOps.invert(img)               # invert image from white/black to black/white
    img = ImageOps.grayscale(img)              # grayscale the image

    img = ImageOps.fit(img, (20,20), Image.ANTIALIAS, bleed=0.06)    # image to 20x20, anti-alias technique and delete unnecessary black pixel at the border

    img2 = ImageOps.fit(img, (28,28))            # image to 28x28

    return img2
