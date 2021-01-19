#!/bin/python3

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input , Dropout, Bidirectional, LSTM, SimpleRNN, Flatten, Dense, MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD, schedules, RMSprop, Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.model_selection import train_test_split
import os
from sys import exit

class BuildModel:
	def __init__(self):
		self.WorkingDir = os.getcwd()
		self.stopping = EarlyStopping(monitor = 'val_loss',
						mode = 'auto',
						restore_best_weights = False,
						baseline = None)

	def load_Data(self):
		if "Images.npy" in os.listdir():
			ImageData = np.load("Images.npy")
			self.X = ImageData[:,:-1]
			self.Y = ImageData[:,-1]
			print (self.X.shape)
			print (self.Y.shape)
		else:
			print ("No data found")
		return 0

	def RNN_Model(self):
		X = self.X.reshape((len(self.X),1,self.X.shape[1]))
		self.Y = to_categorical(self.Y)
		print (X[0].shape)
		print (self.Y.shape)
		print ("========================RNN_MODEL====================================")
		model = Sequential()
		model.add(Input(shape=(1,12288)))

		model.add(LSTM(8,return_sequences=True))
		model.add(BatchNormalization())

		model.add(LSTM(16,return_sequences=True))
		model.add(BatchNormalization())

		model.add(LSTM(64, return_sequences=True))
		model.add(BatchNormalization())

		model.add(Flatten())
		model.add(Dropout(0.3))

		model.add(Dense(64, activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Dense(32, activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Dropout(0.2))

		model.add(Dense(16, activation='sigmoid'))

		model.add(Dense(10, activation='sigmoid'))

		lr_schedule = schedules.ExponentialDecay(
							initial_learning_rate=1e-2,
							decay_steps=500,
							decay_rate=0.45)

		model.compile(optimizer=SGD(learning_rate = lr_schedule), loss = 'categorical_crossentropy', metrics=['accuracy'])
		## 97 epochs are enough for this modal with 90% accuracy
		model.fit(X,self.Y,epochs = 97,validation_split=0.1,batch_size = 64,shuffle = True,verbose=1 ,callbacks = [self.stopping])
		model.save('RNN_Model.h5')
		return 0

	def CNN_Model(self):
		X = self.X.reshape((self.X.shape[0],64,64,3))
		self.Y = to_categorical(self.Y)

		print (X[0].shape)
		print (self.Y[0].shape)
		print ("=====================CNN_MODEL=======================================")
		model = Sequential()
		model.add(Input(shape=(64,64,3)))

		model.add(Conv2D(16,(2,2),activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Conv2D(32,(2,2),activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Conv2D(48,(2,2),activation='relu'))
		model.add(BatchNormalization())

		model.add(Conv2D(64,(2,2),activation='relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dropout(0.4))

		model.add(Dense(64, activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Dense(32, activation='sigmoid'))
		model.add(BatchNormalization())

		model.add(Dropout(0.2))

		model.add(Dense(16, activation='sigmoid'))
		model.add(Dense(10,activation='softmax'))

		model.compile(optimizer = Adam(0.0001),
				loss = 'categorical_crossentropy',
				metrics=['accuracy'])

		## 40 epochs are enough for this model to come up with accuracy=95%
		model.fit(X,self.Y,validation_split=0.2,epochs = 50,batch_size = 64, callbacks = [self.stopping])
		return 0

	def k_neighbor(self,):
		train_X, test_X,train_Y, test_Y = train_test_split(self.X, self.Y,test_size = 0.2)
		KN = kn(n_neighbors = 10,leaf_size = 32,algorithm = 'kd_tree')
		KN.fit(X = train_X,y = train_Y)
		print (KN.score(test_X,test_Y))

bm = BuildModel()
bm.load_Data()
#bm.CNN_Model()
bm.RNN_Model()
#bm.k_neighbor()
