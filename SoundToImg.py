#!/bin/python3

import os
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import re
from sys import exit

class SoundToImg:
	def __init__(self):
		self.WorkingDir = os.getcwd()
		i = 0

	def file_label(self,filename):
		label = -1
		file_labels = re.findall(r'\d+',filename)
		if file_labels:
			label = file_labels[0]
		return int(label)

	def load_Data(self):
		X, Y = [],[]
		files_Directory = os.path.join(self.WorkingDir,'Data/DigitRecordings/')
		filenames = os.listdir(files_Directory)

		for file in filenames:
			filename = os.path.join(files_Directory,file)
			framerate, data = read(filename)  ##for scipy
			X.append(data)
			Y.append(file)

		self.X = X
		self.Y = Y
		self.load_spectrogram()
		return 0

	def load_spectrogram(self):
		spect_X = []
		MAX_LEN_SHAPE = -1
		files_Directory = self.WorkingDir +'/outputs/'
		for audio,filename in zip(self.X,self.Y):
			filename = filename.replace('.wav','.png')
			spect = plt.specgram(audio,Fs = 8000)
			plt.savefig(os.path.join(files_Directory,filename),edgecolor='w')

bm = SoundToImg()
bm.load_Data()
