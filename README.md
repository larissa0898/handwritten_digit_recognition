# deep_learning_project

#### To execute the code in 'mymodel.py', all files have to be in the same folder. The paths of 'train_path' and 'test_path' and of loading and saving the model need to be adapted, when the files were downloaded.
#### The model has already been trained with 60.000 train images and is saved in the file 'model.pt'. It just needs to be loaded into the code again.
#### When testing with the 10.000 test images of the MNIST dataset, the appropriate section of the code ('testing model with test_data') needs to be uncommented and the code can be executed.
#### When testing with my own data, the considered images have to be in the same folder as the 'mymodel.py' file. You can choose beetween 10 images of my own handwriting, 10 images of a third person's handwriting and 10 images of a MNIST-like handwriting, where I tried to imitate the handwriting of most of the MNIST images, because it was obvious that the digits of MNIST have a pretty similar appearance. For example, most of the 'ones' look like this: ![MNIST One](/dl_project/M1.png)

#### While mine look like this: ![My One](/dl_project/L1.png)
#### * When I tested my data, I recognized the following:
                                       * my handwriting: 4/10 images correctly predicted
                                       * third person's handwriting: 3/10 images correctly predicted
                                       * MNIST-like handwriting: 8/10 images correctly predicted

