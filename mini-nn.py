import struct as st
import numpy as np
from matplotlib import pyplot as plt

# Read in all of the training image data

# Files downloaded from: http://yann.lecun.com/exdb/mnist/
filename = {'images': 'data/train-images-idx3-ubyte',
            'labels': 'data/train-labels-idx1-ubyte'}

# Open the training Idx files
train_imagesfile = open(filename['images'], 'rb')
train_labelsfile = open(filename['labels'], 'rb')

# Read the magic number for both files
train_imagesfile.seek(0)
magic_images = st.unpack('>4B', train_imagesfile.read(4))

train_labelsfile.seek(0)
magic_labels = st.unpack('>4B', train_labelsfile.read(4))

# Read in the number of images and their size
num_images = st.unpack('>I', train_imagesfile.read(4))[0]
num_rows = st.unpack('>I', train_imagesfile.read(4))[0]
num_columns = st.unpack('>I', train_imagesfile.read(4))[0]

# Read in the number of labels
num_labels = st.unpack('>I', train_labelsfile.read(4))[0]

# Initialise the arrays for images and labels
images_array = np.zeros((num_images, num_rows, num_columns))
labels_array = np.zeros(num_labels)

# Read in the pixel values
num_bytes = num_images * num_rows * num_columns
images_array = 255 - np.asarray(st.unpack('>' + 'B' * num_bytes,
                                          train_imagesfile.read(num_bytes))).reshape((num_images, num_rows, num_columns))

# Read in the labels
labels_array = np.asarray(
    st.unpack('>' + 'B' * num_labels, train_labelsfile.read(num_labels)))

print(labels_array[0])
plt.imshow(images_array[0], cmap="gray")
plt.show()
