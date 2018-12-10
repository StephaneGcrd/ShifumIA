
# Save augmented images to file

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

from keras import backend as K
import numpy as np
from loaddata import Load
K.set_image_dim_ordering('th')


# load data
name = "images_scissors/"


L = Load(128,128,400)
L.LoadImagesFromFolder(name)
L.createLabels()


X_train = np.expand_dims(L.DATA_X, axis=3)
y_train = L.DATA_LABELS
print(X_train.shape)
# reshape to be [samples][pixels][width][height]

# convert from int to float
X_train = X_train.reshape(X_train.shape[0], 1, 128, 128)
X_train = X_train.astype('float32')
shift = 0.2
# define data preparation
datagen = ImageDataGenerator(
	featurewise_center=True,
	featurewise_std_normalization=True,
	rotation_range=90,
	width_shift_range=shift,
	height_shift_range=shift)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
z = 0
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=400, save_to_dir=name, save_prefix='aug', save_format='png'):
	# create a grid of 3x3 images

	print(z)
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i][0], cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break