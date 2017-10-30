import tensorflow as tf 
from keras import backend as K 
from places205VGG16 import VGG16
from keras.models import Model
import keras.layers 
from keras.objectives import categorical_crossentropy

from sys import path
path.append('/home/longw/Documents/terrainClassification/caffe-tensorflow/examples/imagenet')
#print(sys.path)
from os import listdir
from os.path import isfile, join, basename, normpath, split

from tensorflow.python.platform import gfile
import re
from helper import add_jpeg_decoding, add_input_distortions


from dataset import ImageProducer
from models.helper import DataSpec

import numpy as np
import PIL.Image as Image

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.
  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.
  Returns:
    File system path string to an image that meets the requested parameters.
  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = join(category, '_dir')
  sub_dir = label_lists[sub_dir]
  full_path = os.path.join(sub_dir, base_name)
  return full_path


def load_image( filepath, input_shape ) :
	sub_dirs = [x[0] for x in gfile.Walk(filepath)]
	print(sub_dirs)
	#root directory comes first, so skip
	is_root_dir = True
	extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
	parent_dirs = ('train', 'validation')
	#label_name = ('traversable','non_traversable','climbable')
	result = {}
	training_images = []
	validation_images = []
	training_dir = ''
	validation_dir = ''
	for sub_dir in sub_dirs:
		if is_root_dir == True:
			is_root_dir = False
			continue
		#print(sub_dir)
		parent_dir, dir_name = split(normpath(sub_dir))
		parent_dir = basename(parent_dir)
		#print('parent_dir is '+parent_dir)
		if dir_name == parent_dirs[0] or dir_name == parent_dirs[1]:
			continue
		file_list = []
		tf.logging.info("Looking for images in '" + dir_name + "'")
		for extension in extensions:
			file_glob = join(sub_dir, '*.' + extension)
			file_list.extend(gfile.Glob(file_glob))
		if not file_list:
			tf.logging.warning('No files found')
			continue
		for file_name in file_list:
			base_name = basename(file_name)
			if parent_dir == parent_dirs[0]:
				training_images.append(base_name)
				training_dir = sub_dir
			elif parent_dir == parent_dirs[1]:
				validation_images.append(base_name)
				validation_dir = sub_dir
		label_name = re.sub(r'[^a-z0-9]+', '_', dir_name.lower())
		#print(label_name)
		result[label_name] = {
			'training_dir': training_dir,
			'validation_dir': validation_dir,
			'training': training_images,
			'validation': validation_images,
		}
	print(result)
	return result

	#files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
	#print(files)
 #    img = Image.open(  )
 #    img.load()
 #    img = img.resize(input_shape, Image.HAMMING)
 #    data = np.asarray( img, dtype="int32" )

 #    # image data is now represented as a NumPy array of shape
	# # (inputShape[0], inputShape[1], 3) however we need to expand the
	# # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
	# # so we can pass it through the network
 #    data = np.expand_dims(data, axis=0)
	#return files
	
	# creating the final model 
	#model = Model(inputs = model.input, outputs = predictions)
	#summary = open("model_summary.txt", "w")
	#summary.write(model.summary)
	#summary.close()	

#def display_results

def main():
	input_shape = (224, 224)
	img_path = '/home/longw/Documents/terrainClassification/data/'
	batch_size = 1
	weight_file = 'places205VGG16_weights.npy'
	class_labels = ('climbable', 'traversable', 'non-traversable')
	#train_data_dir = 'data/train'
	#validation_data_dir = 'data/validation'
	
	img_paths = load_image(img_path, input_shape)

	spec = DataSpec(batch_size=batch_size, scale_size=256, crop_size=224, isotropic=True) 

	# placeholder for input data
	input_data = tf.placeholder(tf.float32, [batch_size, spec.crop_size, spec.crop_size, spec.channels]) 

	#weights_data = np.load(weight_file).item()
	# instantiate CNN model
	model = VGG16({'data': input_data})


	# Create an image producer (loads and processes images in parallel)
	#image_producer = ImageProducer(image_paths=img_path, data_spec=spec, batch_size=batch_size)

	#output labels
	labels = tf.placeholder(tf.float32, shape=(None, 3))

	#loss = tf.reduce_mean(categorical_crossentropy(labels, model))

	sess = tf.Session()
	K.set_session(sess)

    # Start the image processing workers
	#coordinator = tf.train.Coordinator()
	#threads = image_producer.start(session=sess, coordinator=coordinator)

	# Set up the image decoding
	jpg_data_tensor, decoded_image_tensor = add_jpeg_decoding(input_shape[0], input_shape[1], 3, 0, 1)

	# Load the converted parameters
	print('Loading the model')
	model.load(weight_file, sess)

	# Load the input image
	print('Loading the images')
	#indices, input_images = image_producer.get(sess)

	output = sess.run(model.get_output(), feed_dict={input_data: input_images})
	#print(output)

	# Stop the worker threads
	#coordinator.request_stop()
	#coordinator.join(threads, stop_grace_period_secs=2)

if __name__ == "__main__":
	main()