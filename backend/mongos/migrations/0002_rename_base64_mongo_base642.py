# Generated by Django 3.2.12 on 2022-04-08 09:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mongos', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='mongo',
            old_name='base64',
            new_name='base642',
        ),
    ]
