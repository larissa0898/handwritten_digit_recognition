# deep_learning_project

## Usage instructions

#### Step 1: 
Download all the files on this GitHub repository.

#### Step 2: 
With 'pip install -r requirements.txt' all relevant libraries are installed. 

![Step1](/dl_project/step1.PNG)

#### Step 3: 
Change the paths in the 'config.ini' file. 

![Step2](/dl_project/step2.PNG)

#### Step 4: 
Now you can run the code in 'testing.py'. You can decide, whether you want to see some of the MNIST images or not and whether you want to train a new model or use an old model.


'Y' for seeing 6 MNIST images. 'n' for no.


'Y' for training a new model. 'n' for loading old model. 

![Step3](/dl_project/step3.PNG)

#### Step 5: 
After that the accuracy of the MNIST test images will be printed in the command prompt as well as my own images with the corresponding predictions of the model and the total accuracy of all my own images. 

![Step4](/dl_project/step4.PNG)

#### Step 6: 
There are two different pre-processing functions ('firstpreprocessing' and 'secondpreprocessing'), which can be used for my own images. Just change it in 'testing.py' in line 121, if you want to use the other pre-processing function.


-->   'firstpreprocessing' yields better results than 'secondpreprocessing', that's why my guess is that my pre-processing is different from the one for the MNIST dataset.

![Step5](/dl_project/step5.PNG)


