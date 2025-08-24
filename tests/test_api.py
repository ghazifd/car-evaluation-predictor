import json
from app import app

def test_predict():
    client = app.test_client()
    payload = {
        "buying": "high",
        "maint": "low",
        "doors": "4",
        "persons": "more",
        "lug_boot": "big",
        "safety": "high"
    }
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "prediction" in data
