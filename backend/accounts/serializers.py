from rest_framework import serializers
from .models import User
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
        model = User
        fields = ('username','password')