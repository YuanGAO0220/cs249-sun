# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR100 small image classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
import tensorflow as tf

def load_data():
  """Loads CIFAR100 dataset.

  Arguments:
      label_mode: one of "fine", "coarse".

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Raises:
      ValueError: in case of invalid `label_mode`.
  """

  dirname = 'cifar-100-python'
  origin = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  fpath = os.path.join(path, 'train')
  train_data, train_label_super = load_batch(fpath, label_key='coarse' + '_labels')
  _ , train_label_sub = load_batch(fpath, label_key='fine' + '_labels')

  fpath = os.path.join(path, 'test')
  test_data, test_label_super = load_batch(fpath, label_key='coarse' + '_labels')
  _ , test_label_sub = load_batch(fpath, label_key='fine' + '_labels')

  train_label_super = np.reshape(train_label_super, (len(train_label_super), 1))
  train_label_sub = np.reshape(train_label_sub, (len(train_label_sub), 1))
  test_label_super = np.reshape(test_label_super, (len(test_label_super), 1))
  test_label_sub = np.reshape(test_label_sub, (len(test_label_sub), 1))

  if K.image_data_format() == 'channels_last':
    train_data = train_data.transpose(0, 2, 3, 1)
    test_data = test_data.transpose(0, 2, 3, 1)


  train_data = train_data.astype(np.float32)
  test_data = test_data.astype(np.float32)
  train_data = train_data / 255
  test_data = test_data /255

  # for i in range(train_data.shape[0]):
  #   print(i)
  #   sess = tf.Session()
  #   with sess.as_default():
  #     train_data[i] = np.asarray(tf.image.per_image_standardization(train_data[i]).eval())
    
  # for i in range(test_data.shape[0]):
  #   sess = tf.Session()
  #   with sess.as_default():
  #     test_data[i] = np.asarray(tf.image.per_image_standardization(test_data[i]).eval())
  # np.save("train_label_super", train_label_super)
  np.save("train_label_sub", train_label_sub)
  # np.save("test_label_super", test_label_super)
  np.save("test_label_sub", test_label_sub)
  # test_data = distorted_inputs(test_data)
  np.save("test_data", test_data)
  #train_data = distorted_inputs(train_data)
  np.save("train_data", train_data)
  


def distorted_inputs(images):
  sess = tf.InteractiveSession()

  for i in range(images.shape[0]):
    if (i % 10 == 0):
      print(i)
    reshaped_image = tf.cast(images[i], tf.float32)

    height = 28
    width = 28

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    images[i] = float_image.eval()



  return images





