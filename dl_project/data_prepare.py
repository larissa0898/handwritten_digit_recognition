import torch
from PIL import Image
from PIL import ImageShow
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import cv2 as cv2
import math
""" 
image = Image.open('1.jpg')
#image = image.imread('1.jpg')

data = np.asarray(image)
#print(type(data))
#print(data.shape)

image2 = Image.fromarray(data)
#print(type(image2))

#print(image2.mode)
#print(image2.size)


#cropped_image = torchvision.transforms.CenterCrop(28)
cropped_image = image2.crop((28, 28, 28, 28))

#print(cropped_image.size)

ImageShow.show(cropped_image)

#pyplot.imshow(cropped_image)
#pyplot.show() """


images = np.zeros((4, 784))
correct_vals = np.zeros((4, 10))

i = 0

for no in [1, 4, 8, 9]:
    img = Image.open(str(no)+".jpg").convert('L')
    img = np.array(img)
    print(img)

    print(img)
    gray = cv2.resize(img, (28, 28))

    #print(type(gray))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(str(no)+".jpg", img)    

    flatten = gray.flatten()/255.0

    images[i] = flatten

    correct_val = np.zeros((10))
    correct_val[no] = 1
    correct_vals[i] = correct_val
    i += 1

#################################################
# np array to tensor to np array to make it visible
####################################################

tensor_array = torch.from_numpy(images)

tensor_array = torch.reshape(tensor_array,(4,28,28))

a = tensor_array.numpy()

""" pyplot.imshow(a[0])
pyplot.show() """


###################################################
# want to fit image into 20x20 pixel box
###################################################
""" 
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

#####################################################
# resize outer box
#######################################################

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows)) """

########################################################
# add missing black rows and columns to get 28x28 pixel
########################################################
""" 
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding), mode='constant')
 """
##############################################################
# get the mass of the image with ndimage
##############################################################
""" 
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)       # ERROR because of missing ndimage module, can't install it
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

################################################################
# shifts image in given directions
###############################################################

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted """




#gray = np.lib.pad(gray,(rowsPadding,colsPadding),mode='constant')

""" shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted """

""" pyplot.imshow(images[0])
pyplot.show()
 """