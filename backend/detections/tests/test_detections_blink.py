import pytest
from .commons import Authentication, get_base64
import time

@pytest.mark.django_db
def test_detections_blink_with_noface(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/blink/'
    data = {
        "blob_data": get_base64.get_blob_to_base64("noface"),
        "count" : 0,
        "total" : 0,
        "time" : 0,
        "detection_flag" : ""
    }

    response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data, content_type="application/json")

    assert response.status_code == 200
    assert response.data == {
    "total": 0,
    "count": 0,
    "res": False,
    "time": 0,
    "detection_flag": "false"
}

@pytest.mark.django_db
def test_detections_blink_with_face(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/blink/'
    data = {
        "blob_data": get_base64.get_blob_to_base64("face"),
        "count" : 0,
        "total" : 0,
        "time" : 0,
        "detection_flag" : ""
    }

    response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data, content_type="application/json")

    assert response.status_code == 200
    assert response.data == {
    "total": 0,
    "count": 0,
    "res": False,
    "time": 500,
    "detection_flag": "detected"
}

@pytest.mark.django_db
def test_detections_blink_with_face_100(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/blink/'
    data = {
        "blob_data": get_base64.get_blob_to_base64("face"),
        "count" : 0,
        "total" : 0,
        "time" : 0,
        "detection_flag" : ""
    }

    start = time.time()
    cnt = 0
    for _ in range(100):
        response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data,
                               content_type="application/json")
        if response.status_code == 200:
            cnt += 1

    elapsed_time = time.time() - start

    assert elapsed_time <= 2.5
    assert cnt == 100
