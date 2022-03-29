from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from .checkpose import check_neck
from .serializers import InitCheckNeckSerializer
from rest_framework.decorators import api_view
import numpy as np

@api_view(['POST'])
def check_neck(request):
    serializer = InitCheckNeckSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):

        # Common
        image_blob = serializer.data.get("blob_data")
        cnt = serializer.data.get('cnt')
        face_string = serializer.data.get("face")
        nose_to_center_string = serializer.data.get('nose_to_center')
        frame = np.fromstring(image_blob, sep='\t')

        # Measurement
        face_mean = serializer.data.get('face_mean')
        nose_mean = serializer.data.get('nose_mean')

        if cnt <= 4:
            if face_string and nose_to_center_string:
                face = face_string[1:-1].split(",")
                nose_to_center = nose_to_center_string[1:-1].split(",")
            else:
                # 기준 값
                face = []  # 얼굴 y축 평균 좌표값
                nose_to_center = []  # 코와 얼굴 중간 간의 거리

            key_points = check_neck.get_keypoints(frame) # 얼굴 좌표들
            _, face_y = check_neck.get_mean_face_positions(key_points)
            distance = check_neck.get_dist_nose_to_face_center(key_points)

            face.append(face_y)
            nose_to_center.append(distance)

            cnt += 1
            data = {
                'face': face,
                'nose_to_center': nose_to_center,
                'cnt': cnt,
            }

            if cnt >= 4:
                face = list(map(float, face))
                nose_to_center = list(map(float, nose_to_center))
                face_mean = np.mean(face)
                nose_mean = np.mean(nose_to_center)
                data[face_mean] = face_mean
                data[nose_mean] = nose_mean

            return Response(data, status=status.HTTP_200_OK)

        else:
            frame = request.data['frame'] # 사용자 화면

            key_points = check_neck.get_keypoints(frame)  # 얼굴 좌표들
            _, face_y = check_neck.get_mean_face_positions(key_points)

            # Y
            y_result = True
            if face_y > face_mean + nose_mean:
                y_result = False # 거북목

            # X
            x_result = True
            angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
            if angle > 100 or angle < 80:
                x_result = False

            data = {
                'y_result': y_result,
                'x_result': x_result
            }

            return Response(data, status=status.HTTP_200_OK)
