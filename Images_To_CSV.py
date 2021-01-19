import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import re
import os

def load_image(location,flatten=True):
	img = None
	try:
		img = Image.open(location).convert('RGB')
		img  = img.resize((64,64))
		img = np.asarray(img,dtype=np.int)
		#plt.imshow(img ,interpolation='nearest')
		#plt.show()
		img = img / 255.0
		if flatten:
			img = img.flatten()
		#print (img.shape)
	except Exception as err:
		print (str(err))
		img = None

	return img

def file_label(filename):
                label = -1
                file_labels = re.findall(r'\d+',filename)
                if file_labels:
                        label = file_labels[0]
                return int(label)

def file_list():
	a = os.listdir('outputs/')
	return a

def main():
	a = file_list()
	Imagedata = []
	for filename in a:
		if not (filename.endswith('.jpeg') or filename.endswith('.png')):
			continue

		filename = ('outputs/%s'%filename)
		data = load_image(filename)
		if data is not None:
			label = file_label(filename)
			if label != -1:
				data = np.append(data,label)
			else:
				continue
		else:
			continue
		Imagedata.append(data)


	#df = pd.DataFrame(Imagedata)
	#df.to_csv("ImageCSV.csv",index=False)
	np.save("Images.npy",Imagedata)
	#print (df.shape)
	print ("Done")

if __name__ == "__main__":
	main()
