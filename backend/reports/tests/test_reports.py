import pytest
from .common import Authentication

@pytest.mark.django_db
def test_get_reports(client):
    access_token = Authentication.signup_login(client)

    url = '/api/reports/'

    response = client.get(url, HTTP_AUTHORIZATION=f'Bearer {access_token}')

    assert response.status_code == 200
    assert response.data == []

@pytest.mark.django_db
def test_post_report(client):
    access_token = Authentication.signup_login(client)

    url = '/api/reports/'

    data = {
        "blink_cnt": 5,
        "neck_cnt": 2,
        "stretching_cnt": 3,
        "start_time": "2203300738",
        "end_time": "2203300938",
        "study_time": "0530"
    }
    response = client.post(url, data=data, HTTP_AUTHORIZATION=f'Bearer {access_token}', content_type="application/json")

    assert response.status_code == 201
    assert response.data == {"id": 1, "blink_cnt": 5, "neck_cnt": 2, "stretching_cnt": 3, "start_time": "2203300738",
                             "end_time": "2203300938", "study_time": "0530"}