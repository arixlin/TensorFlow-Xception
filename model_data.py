#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Author:linming(linming@ztgame.com)
Date: 2017/08/16
"""

import tensorflow as tf
import numpy as np
import os
import math
import glob
import random
from PIL import Image


def save_lable(labelset):
    try:
        np.save('./conf/labelset.npy', labelset)
    except Exception as e:
        raise e


def get_tfimage(image, image_W=208, image_H=208, channelnum=3):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        channelnum: image channel number
    Returns:
        image: 4D tensor [1, width, height, 3], dtype=tf.float32
    """

    # image = tf.image.decode_jpeg(image, channels=channelnum)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, image_W, image_H, channelnum])
    image = tf.cast(image, tf.float32)

    return image


def get_path_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    """
    image_filenames = glob.glob(file_dir + '/*/*')
    image_labels = glob.glob(file_dir + '/*')

    labelset = []
    for labels in image_labels:
        label = labels.split('\\')[1]
        labelset.append(label)

    print(labelset)
    # save_lable(labelset)
    image_list = []
    label_list = []
    for file in image_filenames:
        label = np.argwhere(np.array(labelset) == file.split('\\')[1])[0, 0]
        image_list.append(file)
        label_list.append(label)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()

    np.random.shuffle(temp)

    # print (temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    n_sample = len(label_list)
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = image_list[0:n_train]
    tra_labels = label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = image_list[n_train:-1]
    val_labels = label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


def get_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size,
              capacity, num_classes, label_sytle='sparse', rotate=True):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # image rotate
    if rotate == True:
        # have bugs
        # image = tf.contrib.image.rotate(image, random.uniform(0, 360))
        image = tf.image.rot90(image, k=tf.random_uniform([1, 1], minval=-1, maxval=1, dtype=tf.int32)[0][0])

    # if you want to test the generated batches of images, you might want to comment the following line.
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # image_batch, label_batch = tf.train.shuffle_batch(
    #     [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=3)
    if label_sytle == 'sparse':
        label_batch = tf.reshape(label_batch, [batch_size])
    elif label_sytle == 'dense':
        label_batch = tf.reshape(label_batch, [batch_size])
        label_batch = tf.one_hot(label_batch, depth=num_classes, on_value=1.0,
                                 off_value=0.0, axis=-1, dtype=tf.float32)
    else:
        return None

    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

