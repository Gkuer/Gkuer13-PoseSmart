from django.db import models
from django.conf import settings

# Create your models here.
class Report(models.Model):
    blink_cnt = models.IntegerField()
    neck_cnt = models.IntegerField()
    stretching_cnt = models.IntegerField()
    start_time = models.CharField(max_length=50)
    end_time = models.CharField(max_length=50)
    study_time = models.CharField(max_length=200,null=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
