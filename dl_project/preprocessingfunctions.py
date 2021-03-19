from PIL import Image
from PIL import ImageOps
import cv2 as cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import center_of_mass
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


###########################################################################
# first variant of pre-processing (based on Michael Garris presentation)
###########################################################################

def findCenter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)


def firstpreprocessing(image):
    """first preprocessing function based on the presentation
    of Michael Garris """

    img2 = ~image                # invert image
    img2 = cv2.GaussianBlur(img2,(15,15),0)


    img1 = cv2.imread("blanko.jpg")
    img1 = ~img1                  # invert image

    img2 = cv2.resize(img2, None, fx=0.74, fy=0.74)

    # calculates center of the images
    pt1 = findCenter(img1)
    pt2 = findCenter(img2)

    # calculates the offset
    dx = pt1[0] - int(pt2[0])
    dy = pt1[1] - int(pt2[1])

    # does slice-operation `paste`
    h,w = img2.shape[:2]

    dst = img1.copy()
    dst[dy:dy+h, dx:dx+w] = img2

    # resize image to 28x28
    image = Image.fromarray(dst)
    resized = image.resize((28,28)).convert('L')

    return resized


############################################################
# second variant of pre-processing (based on MNIST website)
############################################################

def secondpreprocessing(image):
    """ second preprocessing function based on
        the MNIST website """

    img = Image.fromarray(image)    

    img = ImageOps.invert(img)               # inverts image from white/black to black/white
    img = ImageOps.grayscale(img)            # grayscale the image

    img = ImageOps.fit(img, (20,20), Image.ANTIALIAS, bleed=0.06)    # image to 20x20, anti-alias technique and delete unnecessary black pixel at the border

    img2 = ImageOps.fit(img, (28,28))          # image to 28x28

    return img2


###############################################################
# shows some of the MNIST images
###############################################################

def show_train_images(train_loader):
    training_data = enumerate(train_loader)
    batch_idx, (train_images, train_labels) = next(training_data)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(train_images[i][0], cmap='gray', interpolation='none')
        plt.title("Label: {}".format(train_labels[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()