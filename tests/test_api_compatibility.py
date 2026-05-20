from fastapi.testclient import TestClient

from analysis.main import app
from analysis.routers import analyze as analyze_router


def test_analyze_text_defaults_input_type_to_text():
    with TestClient(app) as client:
        response = client.post(
            "/analyze/text",
            json={"content": "아토피를 치료합니다."},
        )

    assert response.status_code == 200
    assert response.json()["overall_suspicion_level"] == "의심"


def test_analyze_url_legacy_endpoint(monkeypatch):
    async def fake_extract_from_url(url: str) -> str:
        assert url == "https://example.com/legacy"
        return "아토피를 치료합니다."

    monkeypatch.setattr(analyze_router, "extract_from_url", fake_extract_from_url)

    with TestClient(app) as client:
        response = client.post(
            "/analyze/url",
            json={"content": "https://example.com/legacy"},
        )

    assert response.status_code == 200
    assert response.json()["overall_suspicion_level"] == "의심"


def test_analyze_root_compatibility_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={"content": "피부 보습에 도움을 줍니다."},
        )

    assert response.status_code == 200
    assert response.json()["overall_suspicion_level"] == "정상"
