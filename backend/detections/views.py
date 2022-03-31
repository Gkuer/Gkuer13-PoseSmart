from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from .checkpose import face_landmark
from .serializers import InitCheckNeckSerializer
from rest_framework.decorators import api_view
import numpy as np
import dlib
import base64
import io
from PIL import Image
import cv2 as cv
@api_view(['POST'])
def check_neck(request):

    # JPG TEST 용 (landmark 정상적으로 프린트됨)
    # image = "detections/model/face.jpg"
    # landmark_list = face_landmark.get_landmark(image)
    # print(landmark_list)
    # print("request data: ", type(request.data['face_x']))
    serializer = InitCheckNeckSerializer(data=request.data)

    print(type(request.data['face_x']))
    print(type(request.data['face_x_mean']))
    if serializer.is_valid(raise_exception=True):
        # print('passed')
        # Common
        image_string = serializer.data.get("blob_data")[22:]
        image_data = base64.b64decode(image_string)  
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)

        cnt = serializer.data.get('cnt')
        face_x_string = serializer.data.get("face_x")
        face_y_string = serializer.data.get("face_y")
        nose_to_center_string = serializer.data.get('nose_to_center')

        # Measurement
        face_x_mean = serializer.data.get('face_x_mean')
        face_y_mean = serializer.data.get('face_y_mean')
        nose_mean = serializer.data.get('nose_mean')

        # face landmark list
        landmark_list = face_landmark.get_landmark(img)

        left_eye = landmark_list[42:48]
        right_eye = landmark_list[36:42]

        # x
        right_cheek_x = sum(list(map(lambda x: x[0], landmark_list[0:4]))) / 4
        left_cheek_x = sum(list(map(lambda x: x[0], landmark_list[13:17]))) / 4
        get_face_x = left_cheek_x - right_cheek_x

        # y
        right_eye_y = sum(list(map(lambda x: x[1], right_eye))) / 6
        left_eye_y = sum(list(map(lambda x: x[1], left_eye))) / 6
        nose_y = sum(list(map(lambda x: x[1], landmark_list[31:36]))) / 5

        get_face_y = (right_eye_y + left_eye_y + nose_y) / 3
        dist_nose_to_face_center = abs(nose_y - get_face_y)


        if cnt <= 3:
            if face_y_string and nose_to_center_string and face_x_string:
                face_x = face_x_string.split(",")
                face_y = face_y_string.split(",")
                nose_to_center = nose_to_center_string.split(",")
                print(face_x, face_y)
            else:
                # 기준 값
                face_x = []  # 얼굴 폭
                face_y = []  # 얼굴 y축 평균 좌표값
                nose_to_center = []  # 코와 얼굴 중간 간의 거리
            
            face_x.append(get_face_x)
            face_y.append(get_face_y)
            nose_to_center.append(dist_nose_to_face_center)
            cnt += 1
            data = {
                'face_x': face_x,
                'face_y': face_y,
                'nose_to_center': nose_to_center,
                'cnt': cnt,
                'face_x_mean': 0,
                'face_y_mean' : 0,
                'nose_mean': 0

            }

            if cnt == 4:

                face_x_mean = sum(list(map(float, face_x))) / 5
                face_y_mean = sum(list(map(float, face_y))) / 5
                nose_mean = sum(list(map(float, nose_to_center))) / 5
                print(face_x_mean, 'mean')
                data['face_x_mean'] = face_x_mean
                data['face_y_mean'] = face_y_mean
                data['nose_mean'] = nose_mean

            # data['cnt'] += 1
            print('res: ', data)
            return Response(data, status=status.HTTP_200_OK)

        else:
            
            right_eye_x = sum(list(map(lambda x: x[0], right_eye))) / 6
            left_eye_x = sum(list(map(lambda x: x[0], left_eye))) / 6

            # 얼굴이 내려가거나, 가까워 지는 경우
            y_result = True
            if get_face_y > face_y_mean + nose_mean or face_x_mean * 1.07 <= get_face_x:
                y_result = False

            # 기운 자세의 경우
            x_result = True
            angle = 90 + (np.arctan2(left_eye_y - right_eye_y , left_eye_x - right_eye_x) * 180) / np.pi
            if angle > 100 or angle < 80:
                x_result = False

            data = {
                'y_result': y_result,
                'x_result': x_result
            }
            print('over 4', data)
            return Response(data, status=status.HTTP_200_OK)




def check_blink(request):
    EYE_AR_THRESH = 0.27
    EYE_AR_CONSEC_FRAMES = 2

    # TODO MODEL에 추가해야함
    COUNTER = 0
    TOTAL = 0

    image_string = serializer.data.get("blob_data")[22:]
        image_data = base64.b64decode(image_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    landmark_list = face_landmark.get_landmark(img)
    left_eye = landmark_list[42:48]
    right_eye = landmark_list[36:42]

    # 눈깜박임
    leftEAR = face_landmark.eye_ratio(left_eye)
    rightEAR = face_landmark.eye_ratio(right_eye)

    ear = (leftEAR + rightEAR) / 2.0


    if ear < EYE_AR_THRESH:
        COUNTER += 1

    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL += 1
            print('깜빡', TOTAL)
        COUNTER = 0




