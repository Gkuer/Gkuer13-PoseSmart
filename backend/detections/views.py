from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status

def init_check_neck(request):

    result = True
    data = {
        'result': result,
    }

    return Response(data,status=status.HTTP_200_OK)

def check_neck(request):

    result = True
    data = {
        'result': result,
    }
    return Response(data,status=status.HTTP_200_OK)