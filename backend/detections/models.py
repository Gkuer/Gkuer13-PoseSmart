from django.db import models

# Create your models here.
class Detection(models.Model):
    # blob_data = models.TextField(blank=True, upload_to = 'images/')
    blob_data = models.TextField()
    face_x_mean = models.FloatField(default=0.0)
    face_y_mean = models.FloatField(default=0.0)
    nose_mean = models.FloatField(default=0.0)
    face_x = models.TextField(default=None,null=True, blank=True)
    face_y = models.TextField(default=None,null=True, blank=True)
    nose_to_center = models.TextField(default=None,null=True, blank=True)
    cnt = models.IntegerField(default=0)