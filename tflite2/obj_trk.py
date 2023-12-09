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


import time

import numpy as np
from PIL import Image
import cv2
import math

from initialize_tf import labels, interpreter, input_details, output_details, height, width
from utils import draw_bbox, get_index_of_max_area, inf
from video_capture import cap


display = True
conf_thresh = .6
timeout = 15

# Maximum distance between 2 center points in 2 subsequent frames UNTIL which the object is considered the same
# The value itself is not included, so if distance == dist_thresh, it is not considered the same object
dist_thresh = 10

catched = 0

track_id = 0

pick_x_dist = 10
pick_y_dist = 30

delay_turn_lost = 1


def inf(frame):
    balls = [
    # (ball_type, center_x, center_y, area)
    ]

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
        # Check which objects have confidence > conf_thresh because it's likely that not all the objects detected are above it
        if score > conf_thresh:
            # Get the labels
            object_name = labels[int(classes[0][index])]

            # Get the bounding box coordinates
            y_min = int(max(1, (rects[0][index][0] * height)))
            x_min = int(max(1, (rects[0][index][1] * width)))
            y_max = int(min(height, (rects[0][index][2] * height)))
            x_max = int(min(width, (rects[0][index][3] * width)))

            balls.append((object_name, score, y_min, x_min, y_max, x_max))

            if display:
                draw_bbox(frame, object_name, score, y_min, x_min, y_max, x_max)
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)

    return balls


def enter_room():
    # Robot indietro 3 cm in caso di pallina in entrata!!!!!!!!
    # Rampa subito dietro all'entrata, forse impossibile retrocedere!!!!!!!!!!!
    # Attenzione a non rilevare oggetti al di fuori della stanza!!!!!!!!!!!!!!!!!!!


    # Controllo presenza oggetti in entrata con sensore



    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = cap.read()

    # Resize frame as required by the model
    frame = cv2.resize(frame, (width, height))

    rects, classes, scores = inf()

    # The enumerate() function is used to get both the index and the value of each item in the list
    for index, score in enumerate(scores[0]):
        if score > conf_thresh:
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

            objs[object_name] = (cx, cy)


            if display:
                draw_bbox(frame, y_min, x_min, y_max, x_max, cx, cy, object_name, score)
        
        elif score <= conf_thresh and score > 0:
            # Alzare flag avanti piano e controllare con sensore
            # Anche rilevando un oggetto ce ne potrebbe essere un altro con confidence bassa ignorato


            if False:
                print("Placeholder")


def detect_obj():
    # Initialize max_score
    max_score = 0

    while max_score <= conf_thresh:
        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()

        # Resize frame as required by the model
        frame = cv2.resize(frame, (width, height))

        balls = inf(frame)

        while not balls:
            ret, frame = cap.read()

            # Resize frame as required by the model
            frame = cv2.resize(frame, (width, height))

            balls = inf(frame)

        # Generates a list of all values in the column at the given index, and then finds the maximum value
        max_score = max(row[1] for row in balls)

        if max_score <= conf_thresh:




            # Robot gira di x gradi!!!!!!!!!!!!!!!!!!!
            
            
            
            
            if False:
                print("placeholder")

        if display:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

    return balls


def catch_obj(obj):
    ball_type, score, y_min, x_min, y_max, x_max = obj


    # Distance
    
    
    # Too far
    while (x_max - x_min) > pick_x_dist:


        # Robot indietro!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        # OBJ DETECTION
        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)

        # Resize frame as required by the model
        frame = cv2.resize(frame, (width, height))

        balls = inf(frame)

        # NO BALL DETECTED
        while not balls:
            start_turn = time.time()


            # Start robot gira a dx!!!!!!!!!!!!!!!!!!!!!!!!!!!


            while not balls:
                print('No balls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                if (time.time() - start_turn) >= delay_turn_lost:
                    start_turn = time.time()

                    break
                
                ret, frame = cap.read()

                # Resize frame as required by the model
                frame = cv2.resize(frame, (width, height))

                balls = inf(frame)

            # Start robot gira a sx!!!!!!!!!!!!!!!!!!!!!!!!!!!


            while not balls:
                print('No balls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                if (time.time() - start_turn) >= delay_turn_lost:
                    start_turn = time.time()

                    break
                
                ret, frame = cap.read()

                # Resize frame as required by the model
                frame = cv2.resize(frame, (width, height))

                balls = inf(frame)

        for ball in balls:
            object_name, score, y_min, x_min, y_max, x_max = ball

            draw_bbox(frame, object_name, score, y_min, x_min, y_max, x_max)

        if False:
            cv2.imshow("Frame", frame)

        


    
    
    # Allign

    delta_x = width / 2 - cx


def lost_obj():
    if False:
        print("placeholder")


def main():
    # Implement enter_room!!!!!!!!!!!!!!!!!!!!!!!!!!!


    initial_balls = detect_obj()

    i_max = get_index_of_max_area(initial_balls)

    catch_obj(initial_balls[i_max])




    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
