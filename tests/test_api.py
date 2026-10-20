from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    # Verify the status code
    assert response.status_code == 200

    # Verify the response JSON
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "marketing-sales-predictor-api"


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"tv_budget": 100},
    )

    # Verify the HTTP status code
    assert response.status_code == 200

    # Verify the predicted sales
    data = response.json()
    assert data["predicted_sales"] == pytest.approx(11.7726, abs=0.01)

    # Verify the unit
    assert data["unit"] == "thousand_units"


def test_negative_budget():
    response = client.post(
        "/predict",
        json={"tv_budget": -10},
    )

    # Verify that invalid input is rejected
    assert response.status_code == 422


def test_budget_above_limit():
    response = client.post(
        "/predict",
        json={"tv_budget": 350},
    )

    # Verify HTTP 422
    assert response.status_code == 422

    # Bonus: Verify the validation error refers to tv_budget
    error_details = response.json()["detail"]
    assert any(
        error["loc"][-1] == "tv_budget"
        for error in error_details
    )

def test_missing_budget():
    response = client.post(
        "/predict",
        json={},
    )

    # Verify HTTP 422
    assert response.status_code == 422

    # Bonus: Verify the error indicates the required field is missing
    error_details = response.json()["detail"]
    assert any(
        error["loc"][-1] == "tv_budget"
        and error["type"] == "missing"
        for error in error_details
    )