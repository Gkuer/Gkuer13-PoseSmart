# Generated by Django 4.0.3 on 2022-03-29 09:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detections', '0002_detection_face_detection_nose_to_center'),
    ]

    operations = [
        migrations.AddField(
            model_name='detection',
            name='cnt',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='detection',
            name='face_mean',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='detection',
            name='nose_mean',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='detection',
            name='face',
            field=models.TextField(default=None, null=True),
        ),
        migrations.AlterField(
            model_name='detection',
            name='nose_to_center',
            field=models.TextField(default=None, null=True),
        ),
    ]
