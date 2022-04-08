from rest_framework.response import Response
from rest_framework import status
from .ai_models import face_landmark
from .serializers import CheckNeckSerializer, CheckBlinkSerializer
from rest_framework.decorators import api_view
import numpy as np
import base64
import cv2 as cv
from .tools import detections
from django.contrib.auth import get_user_model


@api_view(['POST'])
def check_neck(request):
    serializer = CheckNeckSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):

        # ================= Common ======================
        # Image
        image_base64 = serializer.data.get("blob_data")[22:]
        image_bytes = base64.b64decode(image_base64)
        image_1darray = np.frombuffer(image_bytes, np.uint8)
        image_3darray = cv.imdecode(image_1darray, cv.IMREAD_COLOR)

        # Origin Landmark
        face_x_str = serializer.data.get("face_x")
        face_y_str = serializer.data.get("face_y")
        nose_to_center_str = serializer.data.get('nose_to_center')

        # Measurement
        face_x_mean = serializer.data.get('face_x_mean')
        face_y_mean = serializer.data.get('face_y_mean')
        nose_mean = serializer.data.get('nose_mean')

        # Flag
        cnt = serializer.data.get('cnt')

        # Face landmark list
        landmark_list = face_landmark.get_landmark(image_3darray)

        # FaceID
        user = get_user_model().objects.get(pk=request.user.pk)
        # ================= Common END ======================

        if landmark_list:  # Find Face Case
            left_eye = landmark_list[42:48]
            right_eye = landmark_list[36:42]

            # X
            right_cheek_x = sum(list(map(lambda x: x[0], landmark_list[0:4]))) / 4
            left_cheek_x = sum(list(map(lambda x: x[0], landmark_list[13:17]))) / 4
            get_face_x = left_cheek_x - right_cheek_x

            # Y
            right_eye_y = sum(list(map(lambda x: x[1], right_eye))) / 6
            left_eye_y = sum(list(map(lambda x: x[1], left_eye))) / 6
            nose_y = sum(list(map(lambda x: x[1], landmark_list[31:36]))) / 5

            get_face_y = (right_eye_y + left_eye_y + nose_y) / 3
            dist_nose_to_face_center = abs(nose_y - get_face_y)

            if cnt <= 3:
                face_x, face_y, nose_to_center = detections.list_from_str(face_x_str, face_y_str, nose_to_center_str)

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
                    'face_y_mean': 0,
                    'nose_mean': 0,
                    'detection_flag': "detected",
                    'face_id_flag': True
                }



                # FaceID Vector Save
                new_vector = face_landmark.get_average_vector(image_3darray)

                IDENTITY_THRESHOLD = 0.45

                standard_vector = list(map(float, user.vector_list[1:-1].split(",")))  # 기존 유저
                distance = np.linalg.norm(np.array(new_vector) - np.array(standard_vector), axis=0)  # 벡터 간 유클리디안 거리 계산

                if new_vector: # 얼굴이 감지되고
                    if (standard_vector == [0 for _ in range(128)]) or (distance < IDENTITY_THRESHOLD): # 최초랑 얼굴이 비교적 가까울때만 학습
                        vector_list = list(map(float, user.vector_list[1:-1].split(",")))
                        vector_cnt = user.vector_cnt
                        new_vector_cnt = vector_cnt + 1

                        vector_list = [i * vector_cnt for i in vector_list]
                        res_vector_list = [(vector_list[i] + new_vector[i]) / new_vector_cnt for i in range(128)]

                        user.vector_list = str(res_vector_list)
                        user.vector_cnt = new_vector_cnt
                        user.save()

                if cnt == 4:
                    data['face_x_mean'] = sum(list(map(float, face_x))) / 4
                    data['face_y_mean'] = sum(list(map(float, face_y))) / 4
                    data['nose_mean'] = sum(list(map(float, nose_to_center))) / 4

            else:
                left_eye_x = sum(list(map(lambda x: x[0], left_eye))) / 6
                right_eye_x = sum(list(map(lambda x: x[0], right_eye))) / 6

                # 기운 자세의 경우
                x_result = True
                angle = 90 + (np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x) * 180) / np.pi
                if angle > 100 or angle < 80:
                    x_result = False

                # 얼굴이 내려가거나, 가까워 지는 경우
                y_result = True

                close = face_x_mean * 1.05 <= get_face_x
                down = get_face_y > (face_y_mean + nose_mean) * 1.02

                if close or down:
                    y_result = False
                    if down and face_x_mean * 0.85 > get_face_x: # [예외처리] : 얼굴이 멀리 가면 거북목이 아니라는 가정 하에 거북목 False 풀어주기
                        y_result = True

                data = {
                    'x_result': x_result,
                    'y_result': y_result,
                    'detection_flag': "detected",
                    'face_id_flag': True
                }

                # FaceID Work
                FACE_THRESHOLD = 0.45

                shapes = face_landmark.get_average_vector(image_3darray) # 새로 들어온 사진
                standard_vector = list(map(float, user.vector_list[1:-1].split(","))) # 기존 유저
                distance = np.linalg.norm(np.array(shapes) - np.array(standard_vector), axis=0)  # 벡터 간 유클리디안 거리 계산

                if distance > FACE_THRESHOLD:
                    data["face_id_flag"] = False

        else: # 얼굴 추적 불가 상태
            face_x, face_y, nose_to_center = detections.list_from_str(face_x_str, face_y_str, nose_to_center_str)

            data = {
                'face_x': face_x,
                'face_y': face_y,
                'nose_to_center': nose_to_center,
                'cnt': cnt,
                'face_x_mean': face_x_mean,
                'face_y_mean': face_y_mean,
                'nose_mean': nose_mean,
                'detection_flag': "false",
                'face_id_flag': True
            }

        return Response(data, status=status.HTTP_200_OK)


@api_view(['POST'])
def check_blink(request):
    EYE_AR_THRESH = 0.27
    serializer = CheckBlinkSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):
        count = serializer.data.get("count")
        total = serializer.data.get("total")
        time = serializer.data.get("time")

        # Image Data Process
        image_base64 = serializer.data.get("blob_data")[22:]
        image_bytes = base64.b64decode(image_base64)
        image_1darray = np.frombuffer(image_bytes, np.uint8)
        image_3darray = cv.imdecode(image_1darray, cv.IMREAD_COLOR)

        landmark_list = face_landmark.get_landmark(image_3darray)

        data = {
            "total": total,
            "count": count,
            "res": False,
            "time": time,
            "detection_flag": "detected"
        }

        if landmark_list:
            left_eye = landmark_list[42:48]
            right_eye = landmark_list[36:42]

            # 눈깜박임
            leftEAR = face_landmark.eye_ratio(left_eye)
            rightEAR = face_landmark.eye_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                data["res"] = True
                data["total"] += 1

            data["time"] += 500

        else:
            data["detection_flag"] = "false"

        return Response(data, status=status.HTTP_200_OK)