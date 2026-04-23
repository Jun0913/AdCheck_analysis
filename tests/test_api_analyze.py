from fastapi.testclient import TestClient

from analysis.main import app
from analysis.routers import analyze as analyze_router


def _post_text(client: TestClient, text: str):
    return client.post(
        "/analyze/text",
        json={
            "input_type": "text",
            "content": text,
        },
    )


def test_health_endpoint_reports_model_status():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "kobert_ready" in payload
    assert "nli_ready" in payload


def test_analyze_text_flags_medical_claim():
    with TestClient(app) as client:
        response = _post_text(client, "이 크림은 아토피를 치료합니다.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["sentence_results"][0]["suspicion_level"] == "의심"


def test_analyze_text_flags_exaggerated_claim():
    with TestClient(app) as client:
        response = _post_text(client, "단 7일 만에 기미 완전 제거, 100% 효과 보장.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["sentence_results"][0]["suspicion_level"] == "의심"


def test_analyze_text_keeps_allowed_moisturizing_claim_normal():
    with TestClient(app) as client:
        response = _post_text(client, "피부 보습에 도움을 주는 크림입니다.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "정상"
    assert payload["sentence_results"][0]["suspicion_level"] == "정상"


def test_analyze_text_does_not_whitelist_doctor_recommendation_claim():
    with TestClient(app) as client:
        response = _post_text(client, "의사가 추천하는 세럼으로 주름이 반드시 사라집니다.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["sentence_results"][0]["suspicion_level"] == "의심"


def test_analyze_text_flags_100_percent_variant_claim():
    with TestClient(app) as client:
        response = _post_text(client, "\uD53C\uBD80 \uAC01\uC9C8 100\uD504\uB85C \uC81C\uAC70")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "\uC758\uC2EC"
    assert payload["sentence_results"][0]["suspicion_level"] == "\uC758\uC2EC"


def test_analyze_text_keeps_caution_for_all_solved_claim():
    with TestClient(app) as client:
        response = _post_text(client, "\uD53C\uBD80 \uBCF4\uC2B5 \uAC74\uC870 \uBAA8\uB450 \uD574\uACB0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "\uC8FC\uC758"
    assert payload["sentence_results"][0]["suspicion_level"] == "\uC8FC\uC758"


def test_analyze_url_uses_extracted_text(monkeypatch):
    async def fake_extract_from_url(url: str) -> str:
        assert url == "https://example.com/ad"
        return "이 크림은 아토피를 치료합니다."

    monkeypatch.setattr(analyze_router, "extract_from_url", fake_extract_from_url)

    with TestClient(app) as client:
        response = client.post(
            "/analyze/text",
            json={
                "input_type": "url",
                "content": "https://example.com/ad",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["original_text"] == "이 크림은 아토피를 치료합니다."


def test_analyze_url_returns_friendly_error(monkeypatch):
    async def fake_extract_from_url(url: str) -> str:
        raise ValueError("크롤링이 불가합니다.")

    monkeypatch.setattr(analyze_router, "extract_from_url", fake_extract_from_url)

    with TestClient(app) as client:
        response = client.post(
            "/analyze/text",
            json={
                "input_type": "url",
                "content": "https://example.com/blocked",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "크롤링이 불가합니다."


def test_analyze_image_uses_ocr_text(monkeypatch):
    monkeypatch.setattr(analyze_router, "extract_from_image", lambda _: "단 7일 만에 기미 완전 제거, 100% 효과 보장.")

    with TestClient(app) as client:
        response = client.post(
            "/analyze/image",
            files={"file": ("sample.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["sentence_results"][0]["suspicion_level"] == "의심"


def test_analyze_image_rejects_non_image_file():
    with TestClient(app) as client:
        response = client.post(
            "/analyze/image",
            files={"file": ("sample.txt", b"not-image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "이미지 파일만 업로드 가능합니다."


def test_analyze_text_does_not_drop_sentence_only_because_ad_filter_is_negative(monkeypatch):
    monkeypatch.setattr(analyze_router, "predict_ad", lambda _: (False, 0.02))
    monkeypatch.setattr(analyze_router, "predict_cosmetic", lambda _: (False, 0.01))

    with TestClient(app) as client:
        response = _post_text(client, "이 크림은 아토피를 치료합니다.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_suspicion_level"] == "의심"
    assert payload["sentence_results"][0]["suspicion_level"] == "의심"
    assert "건너뜀" not in payload["sentence_results"][0]["reason"]
