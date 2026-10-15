import asyncio

import pytest
from starlette.requests import Request
from urllib.parse import parse_qs, urlparse

from app.auth.deps import extract_token, require_staff
from app.auth.jwt import create_access_token
from app.api.routes.auth import (
    build_google_authorization_url,
    create_oauth_state,
    decode_oauth_state,
    hash_password,
    normalize_email,
    validate_password,
    verify_password,
)
from app.core.config import settings


def test_plaintext_seed_password_is_rejected_in_production(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert verify_password("demo-password", "demo-password") is False


def test_plaintext_seed_password_remains_available_for_local_demo(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    assert verify_password("demo-password", "demo-password") is True


def test_signup_password_is_bcrypt_hashed():
    hashed = hash_password("a-secure-password")
    assert hashed != "a-secure-password"
    assert verify_password("a-secure-password", hashed) is True


def test_signup_input_normalization_and_password_policy():
    assert normalize_email("  Person@Example.COM ") == "person@example.com"
    assert validate_password("ten-letters") == "ten-letters"
    with pytest.raises(Exception):
        validate_password("short")


def test_google_oauth_state_is_signed_and_pkce_enabled(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost/callback")
    signed, state, nonce, verifier = create_oauth_state()
    decoded = decode_oauth_state(signed)
    assert decoded["state"] == state
    assert decoded["nonce"] == nonce

    url = build_google_authorization_url(state, nonce, verifier)
    params = parse_qs(urlparse(url).query)
    assert params["response_type"] == ["code"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["scope"] == ["openid email profile"]


def test_customer_role_cannot_pass_staff_dependency():
    with pytest.raises(Exception) as error:
        asyncio.run(require_staff(user={"role": "CUSTOMER"}))
    assert error.value.status_code == 403


def test_staff_session_can_be_read_from_httponly_cookie():
    token = create_access_token(8, 42, "MANAGER", "manager@example.com")
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/v1/auth/me",
        "headers": [(b"cookie", f"{settings.AUTH_COOKIE_NAME}={token}".encode())],
    })

    payload = asyncio.run(extract_token(request, authorization=None))

    assert payload["user_id"] == 8
    assert payload["restaurant_id"] == 42


def test_bearer_clients_remain_supported_without_cookie():
    token = create_access_token(9, 77, "ADMIN", "admin@example.com")
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    payload = asyncio.run(extract_token(request, authorization=f"Bearer {token}"))

    assert payload["restaurant_id"] == 77
