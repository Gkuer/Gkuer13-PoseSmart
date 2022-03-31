from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Detection

class InitCheckNeckSerializer(serializers.ModelSerializer):

    class Meta:
        model = Detection
        fields = ('blob_data', 'face_x', 'face_y', 'nose_to_center', 'cnt', 'face_x_mean', 'face_y_mean', 'nose_mean')