import pytest
from django.urls import reverse
from pytest_drf import APIViewTest, Returns200, UsesGetMethod
from pytest_lambda import lambda_fixture


class TestHelloWorld(
    APIViewTest,
    UsesGetMethod,
    Returns200,
):

    # url = lambda_fixture(lambda: reverse('reports: report_list'))
    @pytest.fixture
    def url(self):
        return reverse('reports:report_list')

    def test_it_returns_hello_world(self, json):
        expected = 'Hello, World!'
        actual = json
        assert expected == actual