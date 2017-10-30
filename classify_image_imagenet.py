# USAGE
# python classify_image.py --image images/soccer_ball.jpg --model vgg16

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense 
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
 	inputShape = (299, 299)
 	preprocess = preprocess_input

train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 2508
nb_validation_samples = 627 
batch_size = 16
epochs = 16

# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet", include_top=False, input_shape= (inputShape[0], inputShape[1], 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:8]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])



# # load the input image using the Keras helper utility while ensuring
# # the image is resized to `inputShape`, the required input dimensions
# # for the ImageNet pre-trained network
# print("[INFO] loading and pre-processing image...")
# image = load_img(args["image"], target_size=inputShape)
# image = img_to_array(image)

# # our input image is now represented as a NumPy array of shape
# # (inputShape[0], inputShape[1], 3) however we need to expand the
# # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# # so we can pass it through thenetwork
# image = np.expand_dims(image, axis=0)

# # pre-process the image using the appropriate function based on the
# # model that has been loaded (i.e., mean subtraction, scaling, etc.)
# image = preprocess(image)



# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
	rescale = 1./255, 
	#horizontal_flip = True, 
	fill_mode = "nearest", 
	#zoom_range = 0.3, 
	#width_shift_range = 0.3, 
	#height_shift_range=0.3, 
	rotation_range=30)

test_datagen = ImageDataGenerator(
	rescale = 1./255)
	# ,
	# horizontal_flip = True,
	# fill_mode = "nearest",
	# zoom_range = 0.3,
	# width_shift_range = 0.3,
	# height_shift_range=0.3,
	# rotation_range=30)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size = inputShape,
	batch_size = batch_size, 
	class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size = inputShape,
	batch_size = batch_size,
	class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=1, mode='auto')

# Train the model 
history = model_final.fit_generator(
	train_generator,
	steps_per_epoch = nb_train_samples // batch_size,
	epochs = epochs,
	validation_data = validation_generator,
	validation_steps = nb_validation_samples // batch_size,
	callbacks = [checkpoint, early])

model.save_weights('second_try_vgg16.h5')

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








# # classify the image
# print("[INFO] classifying image with '{}'...".format(args["model"]))
# preds = model.predict(image)
# P = imagenet_utils.decode_predictions(preds)

# # loop over the predictions and display the rank-5 predictions +
# # probabilities to our terminal
# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
# 	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# # load the image via OpenCV, draw the top prediction on the image,
# # and display the image to our screen
# orig = cv2.imread(args["image"])
# (imagenetID, label, prob) = P[0][0]
# cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)