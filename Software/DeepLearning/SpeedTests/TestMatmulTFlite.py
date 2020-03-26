#!/usr/bin/env python

import numpy as np
import tensorflow as tf

root = tf.train.Checkpoint()
root.f = tf.function(lambda x, y: tf.matmul(x, y))

new_input_data = np.random.randn(2, 100, 100, 3).astype(np.float32)
new_w = np.random.randn(2, 1, 3, 3).astype(np.float32)

input_data = tf.convert_to_tensor(new_input_data)
input_w = tf.convert_to_tensor(new_w)

concrete_func = root.f.get_concrete_function(input_data, input_w)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()