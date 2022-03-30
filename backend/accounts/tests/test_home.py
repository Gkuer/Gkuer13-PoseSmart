import pytest

@pytest.mark.django_db
def test_signup(client):
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/'

    response = client.post(url, data = data)

    assert response.status_code == 201
    assert response.data.get("username") == "testuser"

@pytest.mark.django_db
def test_login(client):
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/login/'

    response = client.post(url, data=data)

    assert response.status_code == 200
    assert type(response.data.get("access")) == str