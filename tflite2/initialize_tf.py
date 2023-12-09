import tensorflow as tf


# Define the variables here
model_file = '/home/giovi/robot/tflite/tflite2/Sample_TFLite_model/detect.tflite'
label_file = '/home/giovi/robot/tflite/tflite2/Sample_TFLite_model/labelmap.txt'
#model_file = 'sartore/razzismo.tflite'
#label_file = 'sartore/labels_map.txt'
num_threads = None


# Load the label map
with open(label_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(
    model_path=model_file,
    num_threads=num_threads)

interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get height and width required for input images
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
