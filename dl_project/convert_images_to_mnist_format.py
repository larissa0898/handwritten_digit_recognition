import os
from PIL import Image
from array import *
from random import shuffle
import numpy as np


# Load from and save to
Names = [['C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project\\training_images','train'], ['C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project\\test_images','test']]

for name in Names:

    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0]): # [1:] Excludes .DS_Store from Mac OS
        path = os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList) # Usefull for further segmenting the validation set

    for filename in FileList:
        filesplit= filename.split("\\")
        lastsplit=filesplit[9].split(".")
        label = int(lastsplit[0])

        Im = Image.open(filename)

        pixel = Im.load()


        width, height = Im.size
        i = 0
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[y,x])
                i +=0

        data_label.append(label) # labels start (one unsigned byte each)

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


    output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    #data_img = bytes(data_image)
    #output_file.write(data_img)
    output_file.close()

    output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files

for name in Names:
    os.system('gzip '+name[1]+'-images-idx3-ubyte')
os.system('gzip '+name[1]+'-labels-idx1-ubyte')




####
#???
####
""" with open('/home/joe/file.txt', 'rb') as f_in, gzip.open('/home/joe/file.txt.gz', 'wb') as f_out:
    f_out.writelines(f_in) """