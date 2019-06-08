# -*- coding: utf-8 -*-
import struct as st
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Read in all of the training image and label data
    # Files downloaded from: http://yann.lecun.com/exdb/mnist/
    filename = {'train_images': 'data/train-images-idx3-ubyte',
                'train_labels': 'data/train-labels-idx1-ubyte'}

    # Open the training Idx files
    train_imagesfile = open(filename['train_images'], 'rb')
    train_labelsfile = open(filename['train_labels'], 'rb')

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
    train_images_array = np.zeros((num_images, num_rows, num_columns))
    train_labels_array = np.zeros(num_labels)

    # Read in the pixel values
    num_bytes = num_images * num_rows * num_columns
    train_images_array = 255 - np.asarray(st.unpack('>' + 'B' * num_bytes,
                                                    train_imagesfile.read(num_bytes))).reshape((num_images, num_rows, num_columns))

    # Read in the labels
    train_labels_array = np.asarray(
        st.unpack('>' + 'B' * num_labels, train_labelsfile.read(num_labels)))

    # Reformat the labels into the expected format e.g label 1 = [0,1,0,0,0,0,0,0,0,0]
    train_output_array = map(
        lambda x: [int(y == x) for y in range(10)], train_labels_array)

    # Reformat the images into the expected input format for the NN i.e concat the rows and cols
    train_input_array = map(np.concatenate, train_images_array)

    print(train_input_array[1])
    print(train_output_array[1])
    plt.imshow(train_images_array[1], cmap="gray")
    plt.show()
