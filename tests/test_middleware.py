"""
Comprehensive tests for rate limiting, API key validation, CORS, abuse detection,
and bot protection middleware.
"""

import os
import time
import pytest
from unittest.mock import patch, AsyncMock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from httpx import AsyncClient, ASGITransport

# Set test env before importing middleware
os.environ["SHRUTI_ENV"] = "production"
os.environ.setdefault("SHRUTI_API_KEYS", "sk-test-alpha:alpha,sk-test-pro:pro")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from shruti.middleware import (
    RateLimitMiddleware,
    ALLOWED_ORIGINS,
    Tier,
    TIER_LIMITS,
    Bucket,
    AbuseTracker,
    validate_api_key,
    reload_api_keys,
    verify_recaptcha,
    RECAPTCHA_PROTECTED_ENDPOINTS,
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal test app
# ---------------------------------------------------------------------------

def create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with the rate limit middleware."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "X-API-Key", "X-ReCAPTCHA-Token", "Content-Type"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Tier", "Retry-After"],
    )

    @app.get("/")
    async def root():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/transcribe")
    async def transcribe():
        return {"result": "transcribed"}

    @app.post("/query")
    async def query():
        return {"answer": "42"}

    @app.get("/stats")
    async def stats():
        return {"count": 0}

    return app


@pytest.fixture
def app():
    reload_api_keys()
    return create_test_app()


@pytest.fixture
def transport(app):
    return ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Unit tests — Bucket
# ---------------------------------------------------------------------------

class TestBucket:
    def test_consume_within_limit(self):
        b = Bucket(tokens=3.0, max_tokens=3)
        assert b.try_consume() is True
        assert b.try_consume() is True
        assert b.try_consume() is True
        assert b.try_consume() is False

    def test_refill_over_time(self):
        b = Bucket(tokens=0.0, max_tokens=3)
        b.last_refill = time.time() - 3600  # 1 hour ago → full refill
        assert b.try_consume() is True

    def test_retry_after(self):
        b = Bucket(tokens=0.0, max_tokens=3)
        b.last_refill = time.time()
        assert b.retry_after > 0


# ---------------------------------------------------------------------------
# Unit tests — AbuseTracker
# ---------------------------------------------------------------------------

class TestAbuseTracker:
    def test_not_blocked_initially(self):
        t = AbuseTracker()
        assert t.is_blocked() is False

    def test_blocks_after_threshold(self):
        t = AbuseTracker()
        for _ in range(10):
            t.record_failure()
        assert t.is_blocked() is True

    def test_unblocks_after_duration(self):
        t = AbuseTracker()
        for _ in range(10):
            t.record_failure()
        t.blocked_until = time.time() - 1  # expired
        assert t.is_blocked() is False


# ---------------------------------------------------------------------------
# Unit tests — API key validation
# ---------------------------------------------------------------------------

class TestAPIKeyValidation:
    def test_valid_alpha_key(self):
        assert validate_api_key("sk-test-alpha") == Tier.ALPHA

    def test_valid_pro_key(self):
        assert validate_api_key("sk-test-pro") == Tier.PRO

    def test_invalid_key(self):
        assert validate_api_key("sk-nonexistent") is None

    def test_reload_keys(self):
        count = reload_api_keys()
        assert count >= 2


# ---------------------------------------------------------------------------
# Integration tests — rate limiting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRateLimiting:
    async def test_anonymous_rate_limit(self, transport):
        """Anonymous users get 3 req/hour on non-exempt endpoints."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(3):
                r = await client.post("/query")
                assert r.status_code == 200, f"Request {i+1} should succeed"
                assert r.headers.get("X-RateLimit-Tier") == "anonymous"

            # 4th request should be rate limited
            r = await client.post("/query")
            assert r.status_code == 429
            body = r.json()
            assert body["tier"] == "anonymous"
            assert "retry_after" in body

    async def test_alpha_key_higher_limit(self, transport):
        """Alpha users get 10 req/hour."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"X-API-Key": "sk-test-alpha"}
            for i in range(10):
                r = await client.post("/query", headers=headers)
                assert r.status_code == 200, f"Request {i+1} should succeed"
                assert r.headers.get("X-RateLimit-Tier") == "alpha"

            r = await client.post("/query", headers=headers)
            assert r.status_code == 429

    async def test_pro_key_high_limit(self, transport):
        """Pro users get 100 req/hour."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer sk-test-pro"}
            for i in range(100):
                r = await client.post("/query", headers=headers)
                assert r.status_code == 200

            r = await client.post("/query", headers=headers)
            assert r.status_code == 429

    async def test_exempt_endpoints_not_limited(self, transport):
        """Health and root endpoints are exempt from rate limiting."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(20):
                r = await client.get("/")
                assert r.status_code == 200
            for _ in range(20):
                r = await client.get("/health")
                assert r.status_code == 200

    async def test_rate_limit_headers(self, transport):
        """Responses include rate limit headers."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/stats")
            assert r.status_code == 200
            assert "X-RateLimit-Limit" in r.headers
            assert "X-RateLimit-Remaining" in r.headers
            assert "X-RateLimit-Tier" in r.headers


# ---------------------------------------------------------------------------
# Integration tests — API key auth
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAPIKeyAuth:
    async def test_invalid_api_key_rejected(self, transport):
        """Invalid API keys get 401."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/query", headers={"X-API-Key": "sk-invalid"})
            assert r.status_code == 401
            assert "Invalid API key" in r.json()["error"]

    async def test_bearer_token_auth(self, transport):
        """Bearer token authentication works."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/query", headers={"Authorization": "Bearer sk-test-alpha"})
            assert r.status_code == 200

    async def test_query_param_auth(self, transport):
        """API key via query parameter works."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/query?api_key=sk-test-pro")
            assert r.status_code == 200
            assert r.headers.get("X-RateLimit-Tier") == "pro"


# ---------------------------------------------------------------------------
# Integration tests — CORS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCORS:
    async def test_allowed_origin(self, transport):
        """Requests from avyay.ai are allowed."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/", headers={"Origin": "https://avyay.ai"})
            assert r.status_code == 200
            assert r.headers.get("access-control-allow-origin") == "https://avyay.ai"

    async def test_shruti_subdomain_allowed(self, transport):
        """Requests from shruti.avyay.ai are allowed."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/", headers={"Origin": "https://shruti.avyay.ai"})
            assert r.status_code == 200
            assert r.headers.get("access-control-allow-origin") == "https://shruti.avyay.ai"

    async def test_disallowed_origin(self, transport):
        """Requests from random origins don't get CORS headers."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/", headers={"Origin": "https://evil.com"})
            assert r.status_code == 200
            # No access-control-allow-origin header for disallowed origin
            assert r.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------------
# Integration tests — abuse detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAbuseDetection:
    async def test_blocks_after_repeated_failures(self, transport):
        """IP gets blocked after repeated invalid API key attempts."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Generate 10 failures with invalid keys
            for i in range(10):
                r = await client.post("/query", headers={"X-API-Key": f"sk-bad-{i}"})
                assert r.status_code == 401

            # Next request should be blocked (403)
            r = await client.post("/query")
            assert r.status_code == 403
            assert "blocked" in r.json()["error"].lower()
            assert "retry_after" in r.json()


# ---------------------------------------------------------------------------
# Unit tests — reCAPTCHA
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReCAPTCHA:
    @patch("shruti.middleware.RECAPTCHA_SECRET", "test-secret")
    async def test_recaptcha_required_when_configured(self, transport):
        """Anonymous users need reCAPTCHA on protected endpoints when configured."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/transcribe")
            assert r.status_code == 403
            assert "reCAPTCHA" in r.json()["error"]

    @patch("shruti.middleware.RECAPTCHA_SECRET", "")
    async def test_recaptcha_skipped_when_not_configured(self, transport):
        """No reCAPTCHA required when secret is not set."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/transcribe")
            assert r.status_code == 200

    async def test_verify_recaptcha_no_secret(self):
        """verify_recaptcha returns True when no secret configured."""
        with patch("shruti.middleware.RECAPTCHA_SECRET", ""):
            valid, score = await verify_recaptcha("any-token")
            assert valid is True

    async def test_verify_recaptcha_success(self):
        """verify_recaptcha passes with valid token."""
        mock_resp = AsyncMock()
        mock_resp.json = lambda: {"success": True, "score": 0.9}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("shruti.middleware.RECAPTCHA_SECRET", "secret"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            valid, score = await verify_recaptcha("good-token")
            assert valid is True
            assert score == 0.9

    async def test_verify_recaptcha_low_score(self):
        """verify_recaptcha fails on low score."""
        mock_resp = AsyncMock()
        mock_resp.json = lambda: {"success": True, "score": 0.1}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("shruti.middleware.RECAPTCHA_SECRET", "secret"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            valid, score = await verify_recaptcha("bot-token")
            assert valid is False
            assert score == 0.1
