# Handwritten Digit Recognition

### Table of Contents
- [Installation](#installation)
- [Usage](#daten)

##Installation: 
Download all the files on this GitHub repository.
With `pip install -r requirements.txt` all relevant libraries are installed. 

## Usage: 
You can run the code in `testing.py`. Decide, whether you want to see some of the MNIST images or not and whether you want to train a new model or use an old model, by choosing:


`Y` for seeing 6 MNIST images. `n` for no.

and

`Y` for training a new model. `n` for loading old model. 

Training takes about 8 minutes.

![Step3](/dl_project/readme_imgs/step3.PNG)


After that the accuracy of the MNIST test images will be printed in the command prompt as well as my own images with the corresponding predictions of the model and the total accuracy of all my own images. 

![Step1.1](/dl_project/readme_imgs/step1.1.PNG)
![Step1.2](/dl_project/readme_imgs/step1.2.PNG)


There are two different pre-processing functions (`firstpreprocessing()` and `secondpreprocessing()`), which can be used for my own images. Just change it in `testing.py` in line 121, if you want to use the other pre-processing function.


-->   `firstpreprocessing()` yields better results than `secondpreprocessing()`

![Step5](/dl_project/readme_imgs/step5.PNG)


