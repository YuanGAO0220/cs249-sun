from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import input
import train

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat

tf.logging.set_verbosity(tf.logging.INFO)

#input.load_data()

train_data = np.load("train_data.npy")
#train_label_super = np.load("train_label_super.npy")

test_data = np.load("test_data.npy")
#test_label_super = np.load("test_label_super.npy")

train_label_sub = np.load("train_label_sub.npy")
test_label_sub = np.load("test_label_sub.npy")
print (train_data.shape)

# Create the Estimator
mnist_classifier = SKCompat(learn.Estimator(
    model_fn=train.cnn_model, model_dir="./model4"))

epoch = 50

for i in range(epoch):

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_label_sub,
      batch_size=128,
      steps=400,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.score(
      x=test_data, 
      y=test_label_sub, 
      metrics=metrics,
      batch_size=128)

  print("************************************************************")
  print("epoch: " + str(i+1))
  print(eval_results)
  print("************************************************************")


#print (train_data.shape)
#print (train_data[1])
#print (test_label_super)
#print (test_label_sub)
#np.savetxt('ytrain1', y_train)