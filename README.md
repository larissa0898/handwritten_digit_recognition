# deep_learning_project

#### Step 1: Download all the files on this GitHub repository.
#### Step 2: With 'pip install -r requirements.txt' all relevant libraries are installed. ![Step1](/dl_project/step1.png)
#### Step 3: Change the paths in the 'config.ini' file. ![Step2](/dl_project/step2.png)
#### Step 4: Now you can run the code in 'testing.py'. You can decide, whether you want to train a new model or use an old model.
####         'Y' for training a new model. 'n' for loading old model. ![Step3](/dl_project/step3.png)
#### Step 5: After that the accuracy of the MNIST test images will be printed in the command propt as well as my own images with the corresponding predictionb of the model and the total accuracy of all my own images in the end. ![Step4](/dl_project/step4.png)
#### Step 6: There are two different preprocessing functions ('firstpreprocessing' and 'secondpreprocessing'), which can be used for my own images. Just change it in 'testing.py' in line 105, if you want to use the other pre-processing function.
                                       * 'firstpreprocessing' yields better results than 'secondpreprocessing', that's why my guess is that my pre-processing is different from the one for the MNIST dataset. ![Step5](/dl_project/step5.png)
#### Step 7: You can also use different images. Just delete the images, which are currently in the same folder ('dl_project') like the '.py' files, and choose another folder ('images1', 'images2', 'images3',...) and copy the images in the same folder ('dl_project') as the '.py' files. Then run the 'testing.py'-file with the prefered preprocessing function. ![Step10](/dl_project/step10.png) ![Step8](/dl_project/step8.png) ![Step9](/dl_project/step9.png)
                                       * By default there are images of the folder 'images2' in the 'dl_project' folder.
