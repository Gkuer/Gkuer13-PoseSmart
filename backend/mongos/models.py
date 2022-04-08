from django.db import models as django_models
from django.conf import settings
from djongo import models
from accounts.models import User
from django.contrib.postgres.fields import ArrayField

class Blob(django_models.Model):
    user_pk = django_models.IntegerField(null=True)
    user_base64 = ArrayField(django_models.TextField(null=True, default=[]))

    class Meta:
        abstract = True

class Mongo(models.Model):
    _id = models.ObjectIdField()
    blob_base64 = models.ArrayField(null=True, model_container=Blob)

    class Meta:
        app_label = 'mongos'