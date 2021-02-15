import os
from PIL import Image
from array import *
from random import shuffle
import numpy as np
import gzip

####################################################################
# load data and save it to
####################################################################

Names = [['C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project\\training_images','train'], ['C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project\\test_images','test']]

for name in Names:

    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0]):
        path = os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList) # Usefull for further segmenting the validation set

    for filename in FileList:
        filesplit= filename.split("\\")
        lastsplit=filesplit[9].split(".")
        label = int(lastsplit[0])

        Img = Image.open(filename)

        pixel = Img.load()


        width, height = Im.size
        i = 0
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[y,x])
                i +=0

        data_label.append(label) # labels start (one unsigned byte each)


####################################################################################
# data to bytes
####################################################################################

    hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))

    data_label = header + data_label

    # additional header for images array

    if max([width,height]) <= 256:
        header.extend([0,0,0,width,0,0,0,height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels')

    header[3] = 3 # Changing MSB for image data (0x00000803)

    data_image = header + data_image

#######################################################################
# save data in .pt and ubyte files
#######################################################################

    pt_file= open(name[1]+'-test.pt', 'wb')
    data_image.tofile(pt_file)

    output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)

    output_file.close()
    pt_file.close()

    pt_file = open(name[1]+'-test.pt','wb')
    data_label.tofile(pt_file)

    output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)

    output_file.close()
    pt_file.close()

""" # gzip resulting files

for name in Names:
    os.system('gzip '+name[1]+'-images-idx3-ubyte')
os.system('gzip '+name[1]+'-labels-idx1-ubyte') """
