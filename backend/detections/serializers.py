from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import NeckDetection, BlinkDetection

class InitCheckNeckSerializer(serializers.ModelSerializer):

    class Meta:
        model = NeckDetection
        fields = ('blob_data', 'face_x', 'face_y', 'nose_to_center', 'cnt', 'face_x_mean', 'face_y_mean', 'nose_mean')

class CheckBlinkSerializer(serializers.ModelSerializer):

    class Meta:
        model = BlinkDetection
        fields = ('blob_data', 'count', 'total', 'time')