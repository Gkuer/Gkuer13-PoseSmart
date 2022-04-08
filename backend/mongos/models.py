from django.db import models

from djongo import models

class Mongo(models.Model):
    test_cnt = models.IntegerField()
    class Meta:
        app_label = 'mongos'