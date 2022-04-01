from django.db import models

# Create your models here.
<<<<<<< HEAD
class Detection(models.Model):
=======
class NeckDetection(models.Model):
    # blob_data = models.TextField(blank=True, upload_to = 'images/')
>>>>>>> 2388f4ca535116e4b728b6fa5604f8887b5b7949
    blob_data = models.TextField()
    face_x_mean = models.FloatField(default=0.0)
    face_y_mean = models.FloatField(default=0.0)
    nose_mean = models.FloatField(default=0.0)
    face_x = models.TextField(default=None,null=True, blank=True)
    face_y = models.TextField(default=None,null=True, blank=True)
    nose_to_center = models.TextField(default=None,null=True, blank=True)
    cnt = models.IntegerField(default=0)

class BlinkDetection(models.Model):
    blob_data = models.TextField()
    count = models.IntegerField(default=0)
    total = models.IntegerField(default=0)
    time = models.IntegerField(default=0) # 1: 1ms