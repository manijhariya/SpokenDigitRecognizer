## Introduction
* created a tool that can Recognize the English Spoken Digits
* Collected data from [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset)
* Engineered Sounds to convert those sound files to images using STFT using matplotlib 
* Engineered sound images to convert those sound files to csv/npy in array format
* Approached for simple K-nearest neighbor approach
* Approached for neural network based solution using CNN (Convolutional neural network) in keras
* Approached for neural network based solution using ANN (Artificial neural newtork) in keras
* Build a client facing API using flask

## Code and Resources Used

Python Version: 3.8
Packages: pandas, numpy, sklearn, matplotlib, flask, tensorflow, scipy

## Data Collecting
Did some research and found the best dataset for spoken digits in [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset)

## Data Cleaning and EDA
After downloading the data, Scripted in python to clean the data so that it can be used by model. Made following changes to clean data.
    * Converted every sound file into its spectrogram using matplotlib and put them in a directory (1500 files)
    * Converted every image into 3 channels (RGB) form.
    * Resized every image into 64 * 64 (W * H)
    * Flatten every image data after taking it into array
    * Saved image array data in csv file for easy to share

## Model Building
Data was ready to fit into the model. Before that i transformed the categorical variables into dummy variables AKA one hot encoded variables
Using Linear Model
    * I tried easy scikit-learn approach with K- nearest neighbor model with train and test split with 0.2 value
    * With K-neighbor model i ended up with 74.4 % accuracy by scoring the model

Using Neural Network
   * Implimented model in Tensorflow2 (keras) in Convolutional Neural Network approach with validation split 0.2 value
   * Compiled Model with Adam Optimizer with learning rate 0.001, EarlyStopping (to overcome Overfitting and underfitting)
   * With CNN model i ended up with 93.2 % accuracy by metrics

   * Implimented Deep Neural Network approach using LSTM (Long Short term memory) layers
   * Compiled model with SGD Optimizer with 0.001 and exponential decay with EarlyStopping
   * With DNN model i ended up with 90 % accuracy.


# Productionization
  In this process, I built a flask API endpoint that was hosted on a local webserver by following along with flask documentation.
  The API endpoint takes in a request in the form of an image of type jpeg, jpg, png and returns the predicted type of Sports ball
  in the image.

# Conclusion
  As you can see Neural Network worked better than k-nearest neighbor algorithm. It is easy to use k-nearest neighbor in image classification
  problem also but with increasing data size it is slow to process as compared to the NN algorithms
