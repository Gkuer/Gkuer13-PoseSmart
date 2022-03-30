import pytest
from .commons import Authentication
import json

@pytest.mark.django_db
def test_get_time(client):
    access_token = Authentication.signup_login(client)

    url = '/api/accounts/time/'

    response = client.get(url, HTTP_AUTHORIZATION=f'Bearer {access_token}')

    assert response.status_code == 200
    assert response.data == {'alarm_sound': 1, 'blink_time': 20, 'neck_time': 20, 'stretching_time': 18000}

@pytest.mark.django_db
def test_set_time(client):
    access_token = Authentication.signup_login(client)

    url = '/api/accounts/time/'

    data = {
        "alarm_sound": 2,
        "blink_time": 40,
        "neck_time": 40,
        "stretching_time": 36000,
    }
    response = client.put(url, data=data, HTTP_AUTHORIZATION=f'Bearer {access_token}', content_type="application/json")

    assert response.status_code == 200
    assert response.data == {"alarm_sound": 2, "blink_time": 40, "neck_time": 40, "stretching_time": 36000}