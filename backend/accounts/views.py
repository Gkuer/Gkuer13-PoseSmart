from django.shortcuts import render, get_object_or_404
from .serializers import AccountSerializer,TimeSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password, check_password

@api_view(['POST','DELETE'])
def accounts(request):
    if request.method == 'POST':
        serializer = AccountSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

    elif request.method == 'DELETE':
        if request.user.username == request.data.get('username') and check_password(request.data.get('password'),encoded=request.user.password):
            user = get_user_model().objects.get(pk=request.user.pk)
            user.delete()
            return Response({"message" : "정상적으로 삭제되었습니다."}, status=status.HTTP_204_NO_CONTENT)
        else:
            data = {
                "errors": "정상적인 삭제 요청이 아닙니다.(사용자 불일치 또는 비밀번호 오류)"
            }
            return Response(data, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT'])
def time(request):
    user = get_user_model().objects.get(pk=request.user.pk)
    if request.method == 'GET':
        blink_time, neck_time, stretching_time = user.blink_time, user.neck_time, user.stretching_time
        data = {
            'blink_time': blink_time,
            'neck_time': neck_time,
            'stretching_time': stretching_time
        }
        return Response(data, status=status.HTTP_200_OK)

    elif request.method == 'PUT':
        serializer = TimeSerializer(instance=user, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)