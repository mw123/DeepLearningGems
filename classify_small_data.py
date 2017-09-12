from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 2508
nb_validation_samples = 627 
batch_size = 32
epochs = 1

### Build the network 
img_input = Input(shape=(256, 256, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

model = Model(inputs = img_input, outputs = x)

model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
print([layer.name for layer in model.layers])

import h5py
weights_path = 'vgg16_1.h5' # ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5)
f = h5py.File(weights_path, 'r')

print(list(f["model_weights"].keys()))

# list all the layer names which are in the model.
layer_names = [layer.name for layer in model.layers]

for i in layer_dict.keys():
    weight_names = f["model_weights"][i].attrs["weight_names"]
    weights = [f["model_weights"][i][j] for j in weight_names]
    index = layer_names.index(i)
    model.layers[index].set_weights(weights)

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import glob

# features = []
# for i in tqdm("images/bmw.png"):
#         im = cv2.imread(i)
#         im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
#         im = np.expand_dims(im, axis =0)
#         outcome = model_final.predict(im)
#         features.append(outcome)

## collect these features and create a dataframe and train a classfier on top of it.
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:
    layer.trainable = False
#model.summary()

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()
# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
	rescale = 1./255, 
	horizontal_flip = True, 
	fill_mode = "nearest", 
	zoom_range = 0.3, 
	width_shift_range = 0.3, 
	height_shift_range=0.3, 
	rotation_range=30)

test_datagen = ImageDataGenerator(
	rescale = 1./255,
	horizontal_flip = True,
	fill_mode = "nearest",
	zoom_range = 0.3,
	width_shift_range = 0.3,
	height_shift_range=0.3,
	rotation_range=30)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size = (img_height, img_width),
	batch_size = batch_size, 
	class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size = (img_height, img_width),
	batch_size = batch_size,
	class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
history = model_final.fit_generator(
	train_generator,
	steps_per_epoch = nb_train_samples // batch_size,
	epochs = epochs,
	validation_data = validation_generator,
	validation_steps = nb_validation_samples // batch_size,
	callbacks = [checkpoint, early])

model.save_weights('first_try.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
