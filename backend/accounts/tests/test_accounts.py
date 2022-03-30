import pytest
from .commons import Authentication

@pytest.mark.django_db
def test_signup(client):
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/'

    response = client.post(url, data=data, content_type="application/json")

    assert response.status_code == 201
    assert response.data.get("username") == "testuser"

@pytest.mark.django_db
def test_login(client):
    # ===== Signup =====
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/'

    client.post(url, data=data, content_type="application/json")

    # ===== END Signup =====

    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/login/'

    response = client.post(url, data=data, content_type="application/json")

    assert response.status_code == 200
    assert type(response.data.get("access")) == str

@pytest.mark.django_db
def test_signout(client):
    access_token = Authentication.signup_login(client)

    url = '/api/accounts/'

    response = client.delete(url, HTTP_AUTHORIZATION=f'Bearer {access_token}')

    assert response.status_code == 204
    assert response.data.get("message") == "정상적으로 삭제되었습니다."
