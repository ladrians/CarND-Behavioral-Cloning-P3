import sys
import os
import time
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import optimizers
from scipy.misc import imsave
import matplotlib.pyplot as plt


# Local data
path1 = 'C:/Proyectos/mooc/carND/3/data/udacity_sample/driving_log.csv'
path2 = 'C:/Proyectos/mooc/carND/3/data/forward_lap/driving_log.csv'
path3 = 'C:/Proyectos/mooc/carND/3/data/backward_lap/driving_log.csv'
path4 = 'C:/Proyectos/mooc/carND/3/data/track2/driving_log.csv'
path5 = 'C:/Proyectos/mooc/carND/3/data/corner_cases/driving_log.csv'

# General parameters
model_name = 'model.h5'
correction = 0.2
verbose = 1 # 1,2
epoch = 8 # 10
split = 0.2
channel, row, col, cols, rows = 1, 160, 320, 32, 32
batch_size = 32
image_choice = 3
dropout = 0.5
debug = False
activation = "elu"
bottom_crop = 20
top_crop = 65
augument_count = 1


def load_data(path):
	'''
	Load images from csv file
	'''
	with open(path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	print("{} lines processed".format(len(lines)))

def process_image(img):
	'''
	return a pipeline for image processing
	Ideas from https://discussions.udacity.com/t/car-gets-out-of-road-in-the-curvy-parts-of-track/587166/2
	'''
	img = hsv_image(img)
	#img = crop_image(img, 0, bottom_crop, 0, top_crop)
	#img = normalize_image(img)
	#img = gray_image(img)
	#img = image_resize(img, row, col)

	return img	

def normalize_image(img):
	'''
	Normalize the current image
	'''
	return img/255.0 - 0.5

def gray_image(img):
	'''
	Convert image to grayscale
	'''
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(label, img):
	'''
	Displays an image, used for debugging
	'''
	cv2.imshow(label, img)
	
def crop_image(img, x, w, y, h):
	'''
	Crop an image, using ideas from https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
	'''
	crop_img = img[h:img.shape[0]-w, 0:img.shape[1]]
	return crop_img	
	
def get_image(path):
	'''
	return an image processed based on a path
	'''
	img = np.asarray(cv2.imread(path))
	img_preprocess = process_image(img)
	return img_preprocess

def debug_image(path):
	'''
	Display the processing pipeline
	'''
	img = np.asarray(cv2.imread(path))
	show_image("Original", img)
	h_img = hsv_image(img)
	show_image("HSV S", h_img)
	crop_img = crop_image(h_img, 0, bottom_crop, 0, top_crop)
	show_image("Crop", crop_img)
	n_img = normalize_image(crop_img)
	show_image("Normalize", n_img)		
	g_img = gray_image(img)
	show_image("Gray", g_img)	
	resize_image = img.resize(1,32,32,1)
	show_image("32x32", g_img)
	img = image_resize(img, row, col)
	show_image("Resize", img)

def get_image_and_measurement(path, angle):
	'''
	return image and measurement
	'''
	return get_image(path), float(angle)

def get_image_flip(img, angle):
	'''
	return an image flipped horizontally
	'''
	flip_img = cv2.flip(img, 1)
	return flip_img, -angle

def hsv_image(img):
	'''
	return an image in S channel of the HSV color space
	'''
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]
	
def get_image_augment_brightness(img):
	'''
	return an image with augmented brightness.
	Ideas taken from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
	'''
	image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	image = np.array(image, dtype = np.float64)
	random_bright = .5+np.random.uniform()
	image[:,:,2] = image[:,:,2]*random_bright
	image[:,:,2][image[:,:,2]>255]  = 255
	image = np.array(image, dtype = np.uint8)
	image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
	return image

def image_resize(img, rows, columns):
	'''
	return an image rezed using the rows as height and columns as width.
	discarded this option
	https://discussions.udacity.com/t/how-to-optimize-the-model-to-reach-the-goal/491414/6
	check issue https://github.com/keras-team/keras/issues/5298
	#return ktf.image.resize_images(img, (r, c))	
	'''
	img_resize = cv2.resize(np.array(img), (columns, rows), interpolation=cv2.INTER_AREA)
	return img_resize
	

def deprocess_image(x):
	'''
	util function to convert a tensor into a valid image
	Taken from https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
	'''
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x	

def get_sample_image(line, angle):
	'''
	return a possible image to the generator
	'''
	rand = np.random.randint(image_choice)
	steer = angle

	if (rand == 0): # Center
		img, steer = get_image_and_measurement(line[0].strip(), steer)
	elif (rand == 1): # Left
		steer = float(steer) + correction
		img, steer = get_image_and_measurement(line[1].strip(), steer)
	elif (rand == 2): # Right
		steer = float(steer) - correction
		img, steer = get_image_and_measurement(line[2].strip(), steer)
	else:
		print("Error, invalid {} value".format(rand))

	if np.random.randint(2) == 0: # Flip image
		img, steer = get_image_flip(img, steer)
		
	return img, steer
	
def generator(samples, batch_size=32):
	'''
	Returns a samples to be processed
	'''
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				angle_original = float(batch_sample[3])
			
				img, angle = get_sample_image(batch_sample, angle_original)
				images.append(img), angles.append(angle)
				'''
				
				img, angle = get_image_and_measurement(batch_sample[0].strip(), angle_original)
				images.append(img), angles.append(angle)
				img, angle = get_image_flip(img, angle)
				images.append(img), angles.append(angle)

				img, angle = get_image_and_measurement(batch_sample[1].strip(), angle_original+correction)
				images.append(img), angles.append(angle)
				img, angle = get_image_flip(img, angle)
				images.append(img), angles.append(angle)
				
				img, angle = get_image_and_measurement(batch_sample[2].strip(), angle_original-correction)
				images.append(img), angles.append(angle)
				img, angle = get_image_flip(img, angle)
				images.append(img), angles.append(angle)			
				'''
				
				
			X_train = np.array(images)
			y_train = np.array(angles)
			# reshape because we are using S from HSV
			X_train = np.reshape(X_train, X_train.shape + (1,))
			y_train = np.reshape(y_train, y_train.shape + (1,))
			yield shuffle(X_train, y_train)

def create_model():
	'''
	Model Architecture based on http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	'''
	model = Sequential()
	# Preprocess
	model.add(Cropping2D(cropping=((top_crop,bottom_crop), (0,0)), input_shape=(row, col, channel)))
	model.add(Lambda(lambda x:x/255.0 - 0.5)) # Normalize data
	# Architecture
	model.add(Conv2D(24,5,5,subsample=(2,2),activation=activation))
	model.add(Conv2D(36,5,5,subsample=(2,2),activation=activation))
	model.add(Conv2D(48,5,5,subsample=(2,2),activation=activation))
	model.add(Conv2D(64,3,3,activation=activation))
	model.add(Conv2D(64,3,3,activation=activation))
	model.add(Dropout(dropout))
	model.add(MaxPooling2D())
	model.add(Dropout(dropout))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def plot_model_history(history_object):
	'''
	Plot the training and validation loss for each epoch
	'''
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show() # block=False
	
def l2_normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

###### Main execution

# Load entries from csv files
lines = []
load_data(path1)
load_data(path2)
load_data(path3)
load_data(path4)
load_data(path5)

'''
# For the documentation only
debug_image(lines[0][0])
cv2.waitKey(0)
sys.exit()
'''

# Load data
train_samples, validation_samples = train_test_split(lines, test_size=split)

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
callbacks = [checkpointer, early_stop]

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model = create_model()

adam = optimizers.Adam(lr=0.001) # default 0.001 tested with values 0.0001, 0.01
model.compile(loss='mse', optimizer=adam) # mse: means square error
history_object = model.fit_generator(train_generator, 
	samples_per_epoch=len(augument_count*train_samples),
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples),
	nb_epoch=epoch,
	verbose=verbose,
	callbacks=callbacks)

model.save(model_name)
print("Saving model as {}".format(model_name))

print("Summary")
print(model.summary())

plot_model_history(history_object)
