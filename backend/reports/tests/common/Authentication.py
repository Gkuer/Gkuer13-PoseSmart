def signup_login(client):
    # ===== Signup =====
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/'

    client.post(url, data=data, content_type="application/json")
    # ===== END Signup =====

    # ===== Login =====
    data = {
        "username": "testuser",
        "password": "12341234"
    }
    url = '/api/accounts/login/'

    response = client.post(url, data=data, content_type="application/json")
    access_token = response.data.get("access")
    # ===== END Login =====

    return access_token