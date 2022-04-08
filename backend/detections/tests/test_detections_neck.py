import pytest
from .commons import Authentication, get_base64
import time

@pytest.mark.django_db
def test_detections_neck_with_noface(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/neck/'
    data = {
        "blob_data": get_base64.get_base64_encoded_blob("noface"),
        "face_x": "1,1",
        "face_y": "11,11",
        "nose_to_center": "11,11",
        "cnt": 1,
        "face_x_mean": 3.0,
        "face_y_mean": 4.0,
        "nose_mean": 5.0,
        "detection_flag": ""
    }

    response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data, content_type="application/json")

    assert response.status_code == 200
    assert response.data == {
    "face_x": [
        "1",
        "1"
    ],
    "face_y": [
        "11",
        "11"
    ],
    "nose_to_center": [
        "11",
        "11"
    ],
    "cnt": 1,
    "face_x_mean": 3.0,
    "face_y_mean": 4.0,
    "nose_mean": 5.0,
    "detection_flag": "false"
}


@pytest.mark.django_db
def test_detections_neck_with_face(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/neck/'
    data = {
        "blob_data": get_base64.get_base64_encoded_blob("face"),
        "face_x": "1,1",
        "face_y": "11,11",
        "nose_to_center": "11,11",
        "cnt": 1,
        "face_x_mean": 3.0,
        "face_y_mean": 4.0,
        "nose_mean": 5.0,
        "detection_flag": ""
    }

    response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data, content_type="application/json")

    assert response.status_code == 200
    assert response.data == {
    "face_x": [
        "1",
        "1",
        76.25
    ],
    "face_y": [
        "11",
        "11",
        107.58888888888889
    ],
    "nose_to_center": [
        "11",
        "11",
        19.011111111111106
    ],
    "cnt": 2,
    "face_x_mean": 0,
    "face_y_mean": 0,
    "nose_mean": 0,
    "detection_flag": "detected"
}

@pytest.mark.django_db
def test_detections_neck_with_face_100(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/neck/'
    data = {
        "blob_data": get_base64.get_base64_encoded_blob("face"),
        "face_x": "1,1",
        "face_y": "11,11",
        "nose_to_center": "11,11",
        "cnt": 1,
        "face_x_mean": 3.0,
        "face_y_mean": 4.0,
        "nose_mean": 5.0,
        "detection_flag": ""
    }

    start = time.time()
    cnt = 0
    for _ in range(100):
        response = client.post(url, HTTP_AUTHORIZATION=f'Bearer {access_token}', data=data, content_type="application/json")
        if response.status_code == 200:
            cnt += 1

    elapsed_time = time.time() - start

    assert elapsed_time <= 2.5
    assert cnt == 100