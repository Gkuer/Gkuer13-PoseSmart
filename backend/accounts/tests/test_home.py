# import pytest
# from django.urls import reverse
# from pytest_drf import APIViewTest, Returns201, UsesPostMethod, AsUser
# from pytest_lambda import lambda_fixture
# from django.contrib.auth import get_user_model
#
# posesmart = lambda_fixture(
#     lambda: get_user_model().objects.create(
#         username='posesmart',
#         password='dj201',
#     ))
#
# class TestReport(
#     APIViewTest,
#     UsesPostMethod,
#     Returns201,
#     AsUser(posesmart)
# ):
#     @pytest.fixture()
#     def url(self):
#         return reverse('accounts:accounts')
#
#     def test_it_returns_201(self, json):
#         expected = 'ff'
#         actual = json
#         assert expected == actual