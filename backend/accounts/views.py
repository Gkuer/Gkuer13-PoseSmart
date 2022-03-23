from django.shortcuts import render, get_object_or_404
from .serializers import AccountSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import User

@api_view(['POST','DELETE'])
def accounts(request):
    if request.method == 'POST':
        serializer = AccountSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

    elif request.method == 'DELETE':
        serializer = AccountSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            # 나중에 get_object_or 404 변환 필요
            if serializer.get_value('username') == request.user.email:
                user = User.objects.get(email=serializer.get_value('username'))
                user.delete()

                return Response(serializer.data, status=status.HTTP_204_NO_CONTENT)
            else:
                data = {
                    "errors": "정상적인 삭제 요청이 아닙니다.(사용자 불일치)"
                }
                return Response(data, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT'])
def time(request):
    user = User.objects.get(pk=request.user.pk)
    if request.method == 'GET':
        blank_time, neck_time, stretching_time = user.blink_time, user.neck_time, user.stretching_time
        data = {
            'blank_time': blank_time,
            'neck_time': neck_time,
            'stretching_time': stretching_time
        }
        return Response(data, status=status.HTTP_200_OK)

    elif request.method == 'PUT':
        serializer = AccountSerializer(instance=user, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)