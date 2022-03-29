from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
import check_neck

def init_check_neck(request):
    # 세션에서 저장된 정보 가져오기?
    face = []   # 얼굴 y축 평균 좌표값
    nose_to_center = []  # 코와 얼굴 중간 간의 거리

    frame = request.data['frame'] # 사용자 화면

    key_points = check_neck.get_keypoints(frame)  # 얼굴 좌표들
    _, face_y = check_neck.get_mean_face_positions(key_points) 
    distance = check_neck.get_dist_nose_to_face_center(key_points)

    face.append(face_y)
    nose_to_center.append(distance)

    # 저장한 data 넘겨주기
    data = {
        'face': face,
        'nose_to_center': nose_to_center
    }

    return Response(data,status=status.HTTP_200_OK)

def check_neck(request):
    frame = request.data['frame'] # 사용자 화면
    # 어떻게 담겨오나요?
    face = request.data['face']
    nose_to_center = request.data['nose_to_center']

    key_points = check_neck.get_keypoints(frame)  # 얼굴 좌표들
    _, face_y = check_neck.get_mean_face_positions(key_points) 
    distance = check_neck.get_dist_nose_to_face_center(key_points)

    result = True
    if face_y > np.mean(face) + np.mean(nose_to_center):
        result = False # 거북목

    angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
    if angle > 100 or angle < 80:
        result = False

    data = {
        'result': result,
    }
    return Response(data,status=status.HTTP_200_OK)
