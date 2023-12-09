# Author: Giovanni Pegoraro
# Date: 06/10/23
# Last modified: --
# Sources:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py
# https://www.comet.com/site/blog/running-tensorflow-lite-image-classification-models-in-python/
# https://github.com/jiteshsaini/robotics-level-4/blob/main/earthrover/object_tracking/object_tracking.py#L90


# TODO tf.while_loop


import time

import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def main():
    # Define the variables here
    video = 'vids/porsche.mp4'
    model_file = 'Sample_TFLite_model/detect.tflite'
    label_file = 'Sample_TFLite_model/labelmap.txt'
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

    # Open video file
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        start_time = time.time()

        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()

        if not ret:
            print('Reached the end of the video!')
            break

        frame = cv2.resize(frame, (width, height))

        # input_details[0]['index'] = the index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], [frame])

        # run the inference
        interpreter.invoke()

        # For TF1 models
        # TFLite_Detection_PostProcess contains the rectangles
        # TFLite_Detection_PostProcess:1 contains the classes for the detected elements
        # TFLite_Detection_PostProcess:2 contains the scores of the rectangles
        # TFLite_Detection_PostProcess:3 contains the total number of detected items

        rects = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])

        # The enumerate() function is used to get both the index and the value of each item in the list
        for index, score in enumerate(scores[0]):
            if score > 0.6:
                object_name = labels[int(classes[0][index])]
                # classes[0][index] look up object name from "labels" array using class index

                # Get the bounding box coordinates
                (y, x, h, w) = rects[0][index]
                # Get the coordinates of the center
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                image = Image.fromarray(frame)
                y_min = int(max(1, (y * image.height)))
                x_min = int(max(1, (x * image.width)))
                y_max = int(min(image.height, (h * image.height)))
                x_max = int(min(image.width, (w * image.width)))

                # draw a rectangle on the image
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

                label = '%s: %d%%' % (object_name, int(score * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(y_min, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (x_min, label_ymin - labelSize[1] - 10),
                              (x_min + labelSize[0], label_ymin + baseLine - 10),
                              (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (x_min, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

        stop_time = time.time()
        fps = 1 / (stop_time - start_time)
        cv2.rectangle(frame, (0, 0), (150, 30), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, 'FPS: ' + str(format(fps, '.2f')), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)  # Draw label text

        cv2.imshow("image", frame)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
