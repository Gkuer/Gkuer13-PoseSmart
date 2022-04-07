# AI 출처
# Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch,
# https://pyimagesearch.com/2018/09/24/opencv-face-recognition/, accessed on 30 March 2022

import dlib
from scipy.spatial import distance as dist
import numpy as np

data_file = "detections/ai_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(data_file)
facerecog = dlib.face_recognition_model_v1('detections/ai_models/dlib_face_recognition_resnet_model_v1.dat') # 얼굴 인식

# Return face landmarks (list)
def get_landmark(frame):
    face_detector = detector(frame, 0)

    for face in face_detector:
        shape = predictor(frame, face)

        landmark_list = []
        for p in shape.parts():
            landmark_list.append([p.x, p.y])

        return landmark_list

# Return EAR
def eye_ratio(eye):
    eye_left_height = dist.euclidean(eye[1], eye[5])
    eye_right_height = dist.euclidean(eye[2], eye[4])

    eye_middle_width = dist.euclidean(eye[0], eye[3])

    EAR = (eye_left_height + eye_right_height) / (2.0 * eye_middle_width)

    return EAR

def get_average_vector(frame):
    face_detector = detector(frame, 0)
    if len(face_detector) == 0:
        return False  # 검출된 얼굴 없는 경우 False 리턴

    shapes = []
    for face in face_detector: # 얼굴이 1개밖에 없다는 가정
        shape = predictor(frame, face)  # 얼굴을 기준으로 랜드마크 좌표가 담긴 object 저장
        shapes.append(shape)

    for shape in shapes:
        return list(facerecog.compute_face_descriptor(frame, shape))