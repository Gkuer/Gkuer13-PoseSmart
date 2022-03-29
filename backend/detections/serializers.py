from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Detection

class InitCheckNeckSerializer(serializers.ModelSerializer):

    class Meta:
        model = Detection
        fields = ('blob_data', 'face', 'nose_to_center', 'cnt', 'face_mean', 'nose_mean')