# AI 출처
# Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch,
# https://pyimagesearch.com/2018/09/24/opencv-face-recognition/, accessed on 30 March 2022

import dlib
from scipy.spatial import distance as dist

data_file = "detections/ai_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(data_file)

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