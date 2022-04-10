from django.db import models as django_models
from djongo import models as djongo_models
from django.contrib.postgres.fields import ArrayField

class Blob(django_models.Model):
    user_pk = django_models.IntegerField(null=True)
    user_base64 = ArrayField(django_models.TextField(null=True, default=[]))

    class Meta:
        abstract = True

class Mongo(djongo_models.Model):
    _id = djongo_models.ObjectIdField()
    blob_base64 = djongo_models.ArrayField(null=True, model_container=Blob)

    class Meta:
        app_label = 'mongos'