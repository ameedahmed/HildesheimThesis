import sys

import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import os

import tensorflow as tf
import tensorflow_hub as hub

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# Import model
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

# Define image shape
IMAGE_SHAPE = (224, 224)

# Configure classifier
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,), trainable=True)
])


# Get labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Load image example
image = Image.open("/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/chih.JPEG")
image = image.resize(IMAGE_SHAPE)
image = np.array(image)/255.0
image = image[:,:,0:3]
plt.imshow(image)

# Class Prediction example
result = classifier.predict(image[np.newaxis, ...])
predicted_class = np.argmax(result)
print('Class: ' + imagenet_labels[predicted_class])

# Create segments

segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

plt.imshow(mark_boundaries(image, segments))
print("Number of superpixels: " + str(len(np.unique(segments))))

"""# CREATE RANDOM CFs - CHANGE UNTIL THE FACTUAL CLASS IS FLIPPED
from isedc.sedc_orig import sedc # SEDC makes that

# Create segments
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

# Try to generate CF
explanation, segments_in_explanation, perturbation, new_class = sedc(image, classifier, segments, 'blur')

# Show explanation image 
plt.imshow(explanation)
print(imagenet_labels[new_class])"""

"""# CREATE TARGETED CF WITH ORIGINAL, NOT OPTIMIZED, FUNCTION
from isedc.sedc_t2 import sedc_t2

# Create segments
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

# Try to generate CF targeting "French bulldog"
explanation, segments_in_explanation, perturbation, new_class = sedc_t2(
    image, 
    classifier, 
    segments, 
    np.where(imagenet_labels=='French bulldog')[0][0], 
    'random')

# Show explanation image
plt.imshow(explanation)
print(imagenet_labels[new_class])"""

# CREATE TARGETED CF WITH OPTIMIZED FUNCTION
from isedc.sedc_t2_fastOG import sedc_t2_fast

# Create segments
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

# Try to generate CF targeting "French bulldog" UNDER 600 SECONDS
explanation, segments_in_explanation, perturbation, new_class = sedc_t2_fast(
    image, 
    classifier, 
    segments, 
    np.where(imagenet_labels=='French bulldog')[0][0], 
    'random', 
    max_time=600)

# Show explanation image
plt.imshow(explanation)
print(imagenet_labels[new_class])

