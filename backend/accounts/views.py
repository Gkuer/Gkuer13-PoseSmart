from django.shortcuts import get_object_or_404
from .serializers import AccountSerializer,TimeSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model

@api_view(['POST','DELETE'])
def accounts(request):
    if request.method == 'POST':
        serializer = AccountSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

    elif request.method == 'DELETE':
        user = get_object_or_404(get_user_model(), pk=request.user.pk)
        user.delete()
        data = {
            "message": "정상적으로 삭제되었습니다."
        }
        return Response(data, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'PUT'])
def time(request):
    user = get_object_or_404(get_user_model(), pk=request.user.pk)
    if request.method == 'GET':
        blink_time, neck_time, stretching_time, alarm_sound = user.blink_time, user.neck_time, user.stretching_time, user.alarm_sound
        data = {
            'blink_time': blink_time,
            'neck_time': neck_time,
            'stretching_time': stretching_time,
            'alarm_sound': alarm_sound,
        }
        return Response(data, status=status.HTTP_200_OK)

    elif request.method == 'PUT':
        serializer = TimeSerializer(instance=user, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)