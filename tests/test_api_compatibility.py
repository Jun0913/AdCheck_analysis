from fastapi.testclient import TestClient

from analysis.main import app


def test_analyze_text_defaults_input_type_to_text():
    with TestClient(app) as client:
        response = client.post(
            "/analyze/text",
            json={"content": "?„í† ?¼ë? ì¹˜ë£Œ?©ë‹ˆ??"},
        )

    assert response.status_code == 200
    assert response.json()["overall_suspicion_level"] == "?˜ì‹¬"


def test_analyze_root_compatibility_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={"content": "?¼ë? ë³´ìŠµ???„ì???ì¤ë‹ˆ??"},
        )

    assert response.status_code == 200
    assert response.json()["overall_suspicion_level"] == "?•ìƒ"


