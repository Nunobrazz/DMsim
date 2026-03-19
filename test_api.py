"""
test_api.py
──────────────────────────────────────────────────────────────────────
Integration tests for the DMsim FastAPI backend.
Uses httpx TestClient + mocked LLM provider to avoid real API calls.

Run: .venv/bin/pytest test_api.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app, PersonaInput, GenerateRequest

client = TestClient(app)


# ── Fixtures ─────────────────────────────────────────────────────────────────

VALID_LLM_RESPONSE = json.dumps({
    "profiles": [
        {
            "name": "Conflicted Council Member",
            "theta_A": 45.0,
            "theta_B": -30.0,
            "p_A": 0.2,
            "p_B": 0.15,
            "rationale": "I hold $RISKY and want it approved."
        },
        {
            "name": "Large Depositor",
            "theta_A": -40.0,
            "theta_B": 35.0,
            "p_A": 0.7,
            "p_B": 0.85,
            "rationale": "I want my deposits safe."
        },
    ]
})

VALID_REQUEST_BODY = {
    "context": "A DAO is voting on collateral approval.",
    "action_a": "Approve $RISKY as collateral",
    "action_b": "Reject $RISKY as collateral",
    "personas": [
        {"name": "Council Member", "description": "Holds $RISKY tokens."},
        {"name": "Depositor", "description": "Has $1M in USDC deposits."},
    ],
}


def _mock_provider():
    """Create a mock LLM provider that returns valid JSON."""
    provider = MagicMock()
    provider.generate.return_value = VALID_LLM_RESPONSE
    return provider, "mock-model"


# ── GET /api/defaults ────────────────────────────────────────────────────────

class TestGetDefaults:
    def test_returns_200(self):
        resp = client.get("/api/defaults")
        assert resp.status_code == 200

    def test_returns_context(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        assert "context" in data
        assert len(data["context"]) > 0

    def test_returns_actions(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        assert "action_a" in data
        assert "action_b" in data
        assert len(data["action_a"]) > 0
        assert len(data["action_b"]) > 0

    def test_returns_personas(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        assert "personas" in data
        assert len(data["personas"]) == 5  # DEFAULT_PERSONAS has 5

    def test_personas_have_name_and_description(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        for persona in data["personas"]:
            assert "name" in persona
            assert "description" in persona
            assert len(persona["name"]) > 0
            assert len(persona["description"]) > 0

    def test_action_a_contains_risky(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        assert "$RISKY" in data["action_a"] or "Approve" in data["action_a"]

    def test_action_b_contains_reject(self):
        resp = client.get("/api/defaults")
        data = resp.json()
        assert "Reject" in data["action_b"] or "collateral" in data["action_b"]


# ── POST /api/generate-profiles — Validation ────────────────────────────────

class TestGenerateValidation:
    def test_empty_body_returns_422(self):
        resp = client.post("/api/generate-profiles", json={})
        assert resp.status_code == 422

    def test_missing_context_returns_422(self):
        body = {**VALID_REQUEST_BODY}
        del body["context"]
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_empty_context_returns_422(self):
        body = {**VALID_REQUEST_BODY, "context": ""}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_missing_action_a_returns_422(self):
        body = {**VALID_REQUEST_BODY}
        del body["action_a"]
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_empty_action_a_returns_422(self):
        body = {**VALID_REQUEST_BODY, "action_a": ""}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_missing_action_b_returns_422(self):
        body = {**VALID_REQUEST_BODY}
        del body["action_b"]
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_empty_personas_returns_422(self):
        body = {**VALID_REQUEST_BODY, "personas": []}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_persona_empty_name_returns_422(self):
        body = {**VALID_REQUEST_BODY, "personas": [{"name": "", "description": "desc"}]}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_persona_empty_description_returns_422(self):
        body = {**VALID_REQUEST_BODY, "personas": [{"name": "Agent", "description": ""}]}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422

    def test_persona_missing_fields_returns_422(self):
        body = {**VALID_REQUEST_BODY, "personas": [{"name": "Agent"}]}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 422


# ── POST /api/generate-profiles — Success ───────────────────────────────────

class TestGenerateSuccess:
    @patch("api.get_provider")
    def test_returns_200_with_profiles(self, mock_get_provider):
        mock_get_provider.return_value = _mock_provider()
        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 200

    @patch("api.get_provider")
    def test_response_contains_profiles_list(self, mock_get_provider):
        mock_get_provider.return_value = _mock_provider()
        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        data = resp.json()
        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert len(data["profiles"]) == 2

    @patch("api.get_provider")
    def test_profile_has_all_fields(self, mock_get_provider):
        mock_get_provider.return_value = _mock_provider()
        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        profile = resp.json()["profiles"][0]
        assert "name" in profile
        assert "theta_A" in profile
        assert "theta_B" in profile
        assert "p_A" in profile
        assert "p_B" in profile
        assert "rationale" in profile

    @patch("api.get_provider")
    def test_profile_values_are_correct(self, mock_get_provider):
        mock_get_provider.return_value = _mock_provider()
        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        profile = resp.json()["profiles"][0]
        assert profile["name"] == "Conflicted Council Member"
        assert profile["theta_A"] == 45.0
        assert profile["theta_B"] == -30.0
        assert profile["p_A"] == 0.2
        assert profile["p_B"] == 0.15
        assert profile["rationale"] == "I hold $RISKY and want it approved."

    @patch("api.get_provider")
    def test_llm_receives_context_and_actions(self, mock_get_provider):
        provider, model = _mock_provider()
        mock_get_provider.return_value = (provider, model)
        client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)

        prompt = provider.generate.call_args[0][0]
        assert "A DAO is voting on collateral approval." in prompt
        assert "Approve $RISKY as collateral" in prompt
        assert "Reject $RISKY as collateral" in prompt

    @patch("api.get_provider")
    def test_llm_receives_persona_descriptions(self, mock_get_provider):
        provider, model = _mock_provider()
        mock_get_provider.return_value = (provider, model)
        client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)

        prompt = provider.generate.call_args[0][0]
        assert "Council Member" in prompt
        assert "Holds $RISKY tokens." in prompt
        assert "Depositor" in prompt

    @patch("api.get_provider")
    def test_single_persona_works(self, mock_get_provider):
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            "profiles": [{"name": "Solo", "theta_A": 10.0, "theta_B": -5.0, "p_A": 0.5, "p_B": 0.6, "rationale": "Only agent."}]
        })
        mock_get_provider.return_value = (provider, "mock")

        body = {**VALID_REQUEST_BODY, "personas": [{"name": "Solo", "description": "A single agent."}]}
        resp = client.post("/api/generate-profiles", json=body)
        assert resp.status_code == 200
        assert len(resp.json()["profiles"]) == 1


# ── POST /api/generate-profiles — Error Handling ────────────────────────────

class TestGenerateErrors:
    @patch("api.get_provider")
    def test_no_api_key_returns_500(self, mock_get_provider):
        mock_get_provider.side_effect = ValueError("Set either GROQ_API_KEY or GOOGLE_API_KEY.")
        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 500
        assert "API_KEY" in resp.json()["detail"]

    @patch("api.get_provider")
    def test_llm_error_returns_502(self, mock_get_provider):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("Model overloaded")
        mock_get_provider.return_value = (provider, "mock")

        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 502
        assert "LLM generation failed" in resp.json()["detail"]

    @patch("api.get_provider")
    def test_invalid_json_from_llm_returns_502(self, mock_get_provider):
        provider = MagicMock()
        provider.generate.return_value = "This is not JSON at all"
        mock_get_provider.return_value = (provider, "mock")

        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 502

    @patch("api.get_provider")
    def test_malformed_profiles_from_llm_returns_502(self, mock_get_provider):
        provider = MagicMock()
        provider.generate.return_value = json.dumps({"profiles": [{"name": "Bad"}]})
        mock_get_provider.return_value = (provider, "mock")

        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 502

    @patch("api.get_provider")
    def test_llm_returns_markdown_wrapped_json(self, mock_get_provider):
        """LLMs sometimes wrap JSON in markdown code blocks."""
        provider = MagicMock()
        provider.generate.return_value = "```json\n" + VALID_LLM_RESPONSE + "\n```"
        mock_get_provider.return_value = (provider, "mock")

        resp = client.post("/api/generate-profiles", json=VALID_REQUEST_BODY)
        assert resp.status_code == 200
        assert len(resp.json()["profiles"]) == 2


# ── CORS ─────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_allows_localhost_3001(self):
        resp = client.options(
            "/api/defaults",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3001"

    def test_cors_allows_127_0_0_1_3001(self):
        resp = client.options(
            "/api/defaults",
            headers={
                "Origin": "http://127.0.0.1:3001",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3001"

    def test_cors_rejects_unknown_origin(self):
        resp = client.options(
            "/api/defaults",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is None
