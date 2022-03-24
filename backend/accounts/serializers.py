from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password

class AccountSerializer(serializers.ModelSerializer):

    def validate_password(self, value: str) -> str:
        """
        Hash value passed by user.

        :param value: password of a user
        :return: a hashed version of the password
        """
        return make_password(value)

    class Meta:
        model = get_user_model()
        fields = ('username','password')

class TimeSerializer(serializers.ModelSerializer):

    class Meta:
        model = get_user_model()
        fields = ('blink_time','neck_time','stretching_time')