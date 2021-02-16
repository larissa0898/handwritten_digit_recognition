import torch
from PIL import Image
import numpy as np
import cv2 as cv2

""" this code prepares images for the model. 
The images have to look like this: 
                                - black background with white digit
                                - have to be named like e.g. 'initial0.jpg
"""

images = []
labels = []
i = 0

for digit in [2]:
    img = cv2.imread("initial{}.jpg".format(digit))       # load image

    blur = cv2.GaussianBlur(img,(5,5),0)               # apply Gaussian Blur

    extract = cv2.selectROI(blur)                         # extract digit with ROI
    imCrop = blur[int(extract[1]):int(extract[1]+extract[3]), int(extract[0]):int(extract[0]+extract[2])]      # save it in variable imCrop

    images.append(imCrop)                        # append image to list
    labels.append(digit)                         # append digit/label to list 
    im = Image.fromarray(images[i])                # convert array to PIL image
    im.save("extract{}.jpg".format(digit))          
    i += 1



################################################################################
# find the center of digit, extract and put it in center of new square image
################################################################################

def findCenter(img):
    print(img.shape, img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)

for m in range(len(labels)):

    img1 = cv2.imread("blanko_.jpg")
    img2 = cv2.resize(cv2.imread("extract{}.jpg".format(labels[m])), None, fx=0.2, fy=0.2)

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

    cv2.imwrite("1_{}.png".format(labels[m]), dst)


#####################################################################
# resize image to 28x28
####################################################################

for img in range(len(labels)):

    image = Image.open("1_{}.png".format(labels[img])).convert('L')          # open image and grayscale it

    resized = image.resize((28,28))

    resized.save("{}.png".format(labels[img]))



############################################################################
# converting ubyte file from convert_images_tomnist_format into .pt file
############################################################################

# image = open("t10k-images-idx3-ubyte")
#torch.save(image, 'C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project\\test_images\\testing\\test.pt')