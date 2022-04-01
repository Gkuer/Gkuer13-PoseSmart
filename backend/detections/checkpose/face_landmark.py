# Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch, 
# https://pyimagesearch.com/2018/09/24/opencv-face-recognition/, accessed on 30 March 2022

import dlib
import numpy as np
import argparse
from scipy.spatial import distance as dist
import cv2 as cv
data_file = "detections/model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(data_file)

# return face landmarks
def get_landmark(frame):
    face_detector = detector(frame, 0)

    for face in face_detector:
        shape = predictor(frame, face)
        # psrint(shape.parts())
        landmark_list = []
        for p in shape.parts():
            landmark_list.append([p.x, p.y])

        return landmark_list
            
# return EAR
def eye_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear