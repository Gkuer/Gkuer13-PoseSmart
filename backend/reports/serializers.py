from rest_framework import serializers
from .models import Report


class ReportListSerializer(serializers.ModelSerializer):

    class Meta:
        model = Report
        exclude = ('user',)


class ReportSerializer(serializers.ModelSerializer):

    class Meta:
        model = Report
        exclude = ('user',)