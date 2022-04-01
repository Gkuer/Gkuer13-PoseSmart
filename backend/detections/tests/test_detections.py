import pytest
from .commons import Authentication, get_base64

@pytest.mark.django_db
def test_detections_neck_with_noface(client):
    access_token = Authentication.signup_login(client)

    url = '/api/detections/'
    print(get_base64.get_blob_to_base64("noface"))
    data = {
        "blob_data": get_base64.get_blob_to_base64("noface"),
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
    assert response.data == {"face_x": "1,1",
        "face_y": "11,11",
        "nose_to_center": "11,11",
        "cnt": 1,
        "face_x_mean": 3.0,
        "face_y_mean": 4.0,
        "nose_mean": 5.0,
        "detection_flag": "false"}

@pytest.mark.django_db
def test_detections_blink(client):
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