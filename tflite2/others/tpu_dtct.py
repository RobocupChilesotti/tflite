# Author: Giovanni Pegoraro
# Date: 06/10/23
# Last modified: --
# Sources:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py
# https://www.comet.com/site/blog/running-tensorflow-lite-image-classification-models-in-python/
# https://github.com/jiteshsaini/robotics-level-4/blob/main/earthrover/object_tracking/object_tracking.py#L90
# https://www.youtube.com/watch?v=GgGro5IV-cs&t=857s


# TODO tf.while_loop
# Try libedgetpu1-max, see here:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Raspberry_Pi_Guide.md


import time

import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import math
import importlib.util
#import tflite_runtime.interpreter as tflite


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def main():
    # Define the variables here
    video = '/home/giovi/robot/tflite/tflite2/vids/juve.mp4'
    model_file = '/home/giovi/robot/tflite/tflite2/Sample_TFLite_model/detect.tflite'
    label_file = '/home/giovi/robot/tflite/tflite2/Sample_TFLite_model/labelmap.txt'
    #model_file = 'sartore/razzismo.tflite'
    #label_file = 'sartore/labels_map.txt'
    num_threads = 14

    # Maximum distance between 2 center points in 2 subsequent frames UNTIL which the object is considered the same
    # The value itself is not included, so if distance == dist_thresh, it is not considered the same object
    dist_thresh = 10

    tracking_objects = {}
    track_id = 0

    # Initialize 'prev_center_pts'
    prev_center_pts = []

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
    
    '''
    interpreter = tf.lite.Interpreter(
        model_path=model_file,
        num_threads=num_threads,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    '''

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

        # Create the list to store the coordinates of the center points of the CURRENT FRAME
        cur_center_pts = []

        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()

        if not ret:
            print('Reached the end of the video!')
            break

        # Resize frame as required by the model
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
                # Get the labels
                object_name = labels[int(classes[0][index])]

                # Get the bounding box coordinates
                y_min = int(max(1, (rects[0][index][0] * height)))
                x_min = int(max(1, (rects[0][index][1] * width)))
                y_max = int(min(height, (rects[0][index][2] * height)))
                x_max = int(min(width, (rects[0][index][3] * width)))

                # Get the coordinates of the center
                # cx = x_min + (x_max - x_min) / 2 = (2 * x_min + x_max - x_min) / 2 = (x_min + x_max) / 2
                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)

                # Add the new coordinates to the list 'center_points'
                cur_center_pts.append((cx, cy))

                # Draw a rectangle on the image
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

                # Draw the label
                label = '%s: %d%%' % (object_name, int(score * 100))  # Example: 'person: 72%'
                # Get font details
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window
                label_ymin = max(y_min, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(frame, (x_min, label_ymin - labelSize[1] - 10),
                              (x_min + labelSize[0], label_ymin + baseLine - 10),
                              (255, 255, 255), cv2.FILLED)
                # Draw label text
                cv2.putText(frame, label, (x_min, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2)

                # Draw the center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        '''
        # Compare each center point from the current frame with each center point with the previous frame
        for pt in cur_center_pts:
            for pt2 in prev_center_pts:
                # Calculate the distance
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < dist_thresh:
                    tracking_objects[track_id] = pt
                    track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        '''

        # Break the loop on 'q' key press
        if cv2.waitKey(25) == ord('q'):
            break

        # Save current center points for the comparison with the next frame
        prev_center_pts = cur_center_pts.copy()

        # Calculate fps
        stop_time = time.time()
        fps = 1 / (stop_time - start_time)

        # Display fps
        cv2.rectangle(frame, (0, 0), (150, 30), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, 'FPS: ' + str(format(fps, '.2f')), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)  # Draw label text

        cv2.imshow("image", cv2.resize(frame, (720, 480)))

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
