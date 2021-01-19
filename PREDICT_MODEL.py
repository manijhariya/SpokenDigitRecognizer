#!/bin/python3

from tensorflow.keras.models import load_model
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

class PREDICT_MODEL:
	def __init__(self,):
		self.model = load_model("RNN_Model.h5")

	def Predict(self,location):
		fr, data = read(location)
		filename = filename.replace('.wav','.png')
		plt.specogram(data, Fs=fr)
		plt.savefg(filename,edgecolor = 'w')
		img = load_image(filename)
		img = np.reshape((1,1,12288))
		result = self.model.predict(img)
		result = np.argmax(result[0])
		return (result)
