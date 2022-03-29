from django.db import models

# Create your models here.
class Detection(models.Model):
    blob_data = models.TextField()
    face_mean = models.FloatField(default=0.0)
    nose_mean = models.FloatField(default=0.0)
    face = models.TextField(default=None)
    nose_to_center = models.TextField(default=None)
    cnt = models.IntegerField(default=0)