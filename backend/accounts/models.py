from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    blink_time = models.IntegerField(default=20)
    neck_time = models.IntegerField(default=20)
    stretching_time = models.IntegerField(default=18000)
    alarm_sound = models.IntegerField(default=1)