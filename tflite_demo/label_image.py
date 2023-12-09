# Author: Giovanni Pegoraro
# Date: 06/10/23
# Last modified: --
# Sources:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py
# https://www.comet.com/site/blog/running-tensorflow-lite-image-classification-models-in-python/
import time

import numpy as np
from PIL import Image
import tensorflow as tf


# Define the variables here
image = 'grace_hopper.bmp'
model_file = 'Sample_TFLite_model/detect.tflite'
label_file = 'Sample_TFLite_model/labelmap.txt'
input_mean = 127.5
input_std = 127.5
num_threads = None
A_ext_delegate = None
A_ext_delegate_options = None

min_conf_threshold = 0.5


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
  ext_delegate = None
  ext_delegate_options = {}

  '''
  # parse extenal delegate options
  if A_ext_delegate_options is not None:
    options = A_ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if (len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()
      else:
        raise RuntimeError('Error parsing delegate option: ' + o)

  # load external delegate
  if A_ext_delegate is not None:
    print('Loading external delegate from {} with args: {}'.format(
        A_ext_delegate, ext_delegate_options))
    ext_delegate = [
        tflite.load_delegate(A_ext_delegate, ext_delegate_options)
    ]
  '''

  interpreter = tf.lite.Interpreter(
      model_path=model_file,
      experimental_delegates=ext_delegate,
      num_threads=num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  '''
  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]'''
  #D labels is compatible
  labels = load_labels(label_file)

  #D Added
  # Have to do a weird fix for label map if using the COCO "starter model" from
  # https://www.tensorflow.org/lite/models/object_detection/overview
  # First label is '???', which has to be removed.
  if labels[0] == '???':
    del (labels[0])
  '''
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
  '''
  boxes_idx, classes_idx, scores_idx = 1, 3, 0

  # Retrieve detection results
  boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
  classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
  scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

  # Loop over all detections and draw detection box if confidence is above minimum threshold
  for i in range(len(scores)):
    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
      # Get bounding box coordinates and draw box

      # Draw label
      object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
      label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'