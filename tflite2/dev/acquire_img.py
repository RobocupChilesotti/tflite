import cv2
from initialize_tf import height, width

video = '/home/giovi/robot/tflite/tflite2/vids/walk.mp4'

def initialize_stream():
    # Open video file
    cap = cv2.VideoCapture(video)

    return cap

def get_img(cap):
    ret, img = cap.read()

    img = cv2.resize(img, (width, height))

    return img
