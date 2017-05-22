from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

NUM_CLASSES = 20

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def cnn_model(features, labels, mode):
  # """Model function for CNN."""
  # # Input Layer
  # # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # # MNIST images are 28x28 pixels, and have one color channel
  # input_layer = tf.reshape(features, [-1, 32, 32, 3])

  # # Convolutional Layer #1
  # # Computes 32 features using a 5x5 filter with ReLU activation.
  # # Padding is added to preserve width and height.
  # # Input Tensor Shape: [batch_size, 28, 28, 1]
  # # Output Tensor Shape: [batch_size, 28, 28, 32]
  # conv1 = tf.layers.conv2d(
  #   inputs=input_layer,
  #   filters=32,
  #   kernel_size=[5, 5],
  #   padding="same",
  #   activation=tf.nn.relu,
  #   bias_initializer = tf.constant_initializer(0.0))

  # # Pooling Layer #1
  # # First max pooling layer with a 2x2 filter and stride of 2
  # # Input Tensor Shape: [batch_size, 28, 28, 32]
  # # Output Tensor Shape: [batch_size, 14, 14, 32]
  
  # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

  # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

  # # Convolutional Layer #2
  # # Computes 64 features using a 5x5 filter.
  # # Padding is added to preserve width and height.
  # # Input Tensor Shape: [batch_size, 14, 14, 32]
  # # Output Tensor Shape: [batch_size, 14, 14, 64]
  # conv2 = tf.layers.conv2d(
  #   inputs=norm1,
  #   filters=32,
  #   kernel_size=[5, 5],
  #   padding="same",
  #   activation=tf.nn.relu,
  #   bias_initializer = tf.constant_initializer(0.1))

  # # Pooling Layer #2
  # # Second max pooling layer with a 2x2 filter and stride of 2
  # # Input Tensor Shape: [batch_size, 14, 14, 64]
  # # Output Tensor Shape: [batch_size, 7, 7, 64]

  # #norm1 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

  # conv3 = tf.layers.conv2d(
  #   inputs=pool2,
  #   filters=64,
  #   kernel_size=[5, 5],
  #   padding="same",
  #   activation=tf.nn.relu,
  # 	bias_initializer = tf.constant_initializer(0.1))

  # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

  # # Flatten tensor into a batch of vectors
  # # Input Tensor Shape: [batch_size, 7, 7, 64]
  # # Output Tensor Shape: [batch_size, 7 * 7 * 64]

  # pool2_flat = tf.reshape(pool3, [-1, 3 * 3 * 64])

  # # Dense Layer
  # # Densely connected layer with 1024 neurons
  # # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # # Output Tensor Shape: [batch_size, 1024]
  # dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)

  # dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.relu)

  # # Add dropout operation; 0.6 probability that element will be kept
  # dropout = tf.layers.dropout(
  #   inputs=dense2, rate=0.1, training=mode == learn.ModeKeys.TRAIN)

  # # Logits layer
  # # Input Tensor Shape: [batch_size, 1024]
  # # Output Tensor Shape: [batch_size, 10]
  # logits = tf.layers.dense(inputs=dropout, units=20)
  


  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(features, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 8*8*64
    reshape = tf.reshape(pool2, [-1, dim])
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('logits') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # lr = tf.train.exponential_decay(0.1,
  #                                 tf.contrib.framework.get_global_step(),
  #                                 decay_steps = 4000,
  #                                 decay_rate = 0.1,
  #                                 staircase=True)

  lr_decay_fn = lambda lr,global_step : tf.train.exponential_decay(lr, global_step, 4000, 0.2, staircase=False)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.1,
      learning_rate_decay_fn = lr_decay_fn,
      optimizer="SGD")

  # Generate Predictions
  predictions = {
  "classes": tf.argmax(
    input=logits, axis=1),
  "probabilities": tf.nn.softmax(
    logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
    mode=mode, predictions=predictions, loss=loss, train_op=train_op)