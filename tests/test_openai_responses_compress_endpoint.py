from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy, create_app


def _responses_payload() -> dict[str, Any]:
    return {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "raw tool output " * 20,
            }
        ],
        "stream": True,
    }


def test_responses_compress_requires_request_object() -> None:
    app = create_app(ProxyConfig())

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses/compress",
            json={"model": "gpt-5"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Missing required object field: request"


def test_responses_compress_rejects_invalid_token_budget() -> None:
    app = create_app(ProxyConfig())

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses/compress",
            json={
                "request": _responses_payload(),
                "model": "gpt-5",
                "token_budget": 0,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "token_budget must be a positive integer"


def test_responses_compress_returns_metrics_without_upstream(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_compress(
        self: HeadroomProxy,
        payload: dict[str, Any],
        *,
        model: str,
        request_id: str,
        token_budget: int | None = None,
    ) -> tuple[dict[str, Any], bool, int, list[str], str | None, int, int, int, dict[str, float]]:
        captured["payload"] = payload
        captured["model"] = model
        captured["request_id"] = request_id
        captured["token_budget"] = token_budget
        compressed = dict(payload)
        compressed["input"] = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "compressed output",
            }
        ]
        return (
            compressed,
            True,
            40,
            ["openai:responses:test"],
            None,
            1000,
            400,
            100,
            {},
        )

    async def fail_upstream(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("upstream must not be called")

    monkeypatch.setattr(
        HeadroomProxy,
        "_compress_openai_responses_payload_in_executor",
        fake_compress,
    )
    monkeypatch.setattr(HeadroomProxy, "_retry_request", fail_upstream, raising=False)

    app = create_app(ProxyConfig())
    payload = _responses_payload()

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses/compress",
            json={
                "request": payload,
                "model": "gpt-5",
                "token_budget": 123,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["request"]["input"][0]["output"] == "compressed output"
    assert body["tokens_before"] == 100
    assert body["tokens_after"] == 60
    assert body["tokens_saved"] == 40
    assert body["compression_ratio"] == 0.6
    assert body["transforms_applied"] == ["openai:responses:test"]
    assert captured["payload"] == payload
    assert captured["model"] == "gpt-5"
    assert captured["token_budget"] == 123
