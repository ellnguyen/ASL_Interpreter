# ASL_Interpreter

## Team members:
Ellie Nguyen, Elle Sanchez, Gus Mitkus, Heidi Wakefield

## Class: SCI 200 Grand Challenges in Science and Engineering

## Project:
This repository contains all the files necessary to load a CNN model we created in Google Colab and use it to classify sign language letters from the webcam.    

The repo contains a python script named "ASLInterpreter.py"    
The script will 
- load the image classification CNN model file called "asl_model.h5"
- load the labels from a file called "label_code.txt"
- capture images from the camera, convert them into normalized numpy arrays, and use the model to predict the image
- print the class of the image with a confidence score


## Files 
1. ASLInterpreter.py
2. asl_model.h5
3. label_code.txt

## Known Errors 

## References
### Train and test datasets taken from Kaggle
https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data

### Exporting models from google colab
https://colab.research.google.com/github/keras-team/autokeras/blob/master/docs/ipynb/export.ipynb

### Handwritten Digital Classification CNN
Handwritten_Digital_Classification_CNN
https://colab.research.google.com/drive/12VSE_s8oo564mhlb5h9O5MvL8VwxbzI9#scrollTo=9ulaiaROGNMJ


## Instructions
1. Download the asl_model.h5, label_code.txt, and ASLInterpreter.py files
2. Make sure they are all in the same folder before running:  
python ASLInterpreter.py
3. To exit, select the window displaying the webcam input and press esc on the keyboard

