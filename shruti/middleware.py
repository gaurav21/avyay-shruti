"""
Rate limiting, API key validation, bot protection, and abuse detection middleware for ŚRUTI API.
"""

import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import httpx
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    ANONYMOUS = "anonymous"
    ALPHA = "alpha"
    PRO = "pro"


TIER_LIMITS: Dict[Tier, int] = {
    Tier.ANONYMOUS: 3,     # 3 req/hour
    Tier.ALPHA: 10,        # 10 req/hour
    Tier.PRO: 100,         # 100 req/hour
}

TIER_WINDOW = 3600  # 1 hour in seconds


# ---------------------------------------------------------------------------
# In-memory token bucket (per-key or per-IP)
# ---------------------------------------------------------------------------

@dataclass
class Bucket:
    tokens: float
    max_tokens: int
    last_refill: float = field(default_factory=time.time)

    def try_consume(self) -> bool:
        """Refill tokens based on elapsed time, then try to consume one."""
        now = time.time()
        elapsed = now - self.last_refill
        refill = (elapsed / TIER_WINDOW) * self.max_tokens
        self.tokens = min(self.max_tokens, self.tokens + refill)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    @property
    def retry_after(self) -> int:
        """Seconds until next token is available."""
        if self.tokens >= 1:
            return 0
        tokens_needed = 1 - self.tokens
        return int((tokens_needed / self.max_tokens) * TIER_WINDOW) + 1


# ---------------------------------------------------------------------------
# API key store — integrates with Avyay alpha signup system
# ---------------------------------------------------------------------------

# Environment-driven valid keys (comma-separated)
# Format: key:tier  e.g. "sk-abc123:alpha,sk-pro456:pro"
_API_KEY_STORE: Optional[Dict[str, Tier]] = None


def _load_api_keys() -> Dict[str, Tier]:
    """Load API keys from environment.  Keys can come from:
    - SHRUTI_API_KEYS: comma-separated key:tier pairs
    - SHRUTI_ALPHA_KEYS: comma-separated keys (all alpha tier)
    - SHRUTI_PRO_KEYS: comma-separated keys (all pro tier)
    """
    global _API_KEY_STORE
    if _API_KEY_STORE is not None:
        return _API_KEY_STORE

    store: Dict[str, Tier] = {}

    # Generic key:tier pairs
    raw = os.getenv("SHRUTI_API_KEYS", "")
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, tier_str = pair.rsplit(":", 1)
            try:
                store[key.strip()] = Tier(tier_str.strip().lower())
            except ValueError:
                logger.warning(f"Unknown tier '{tier_str}' for key, defaulting to alpha")
                store[key.strip()] = Tier.ALPHA

    # Convenience env vars
    for key in os.getenv("SHRUTI_ALPHA_KEYS", "").split(","):
        key = key.strip()
        if key:
            store[key] = Tier.ALPHA

    for key in os.getenv("SHRUTI_PRO_KEYS", "").split(","):
        key = key.strip()
        if key:
            store[key] = Tier.PRO

    _API_KEY_STORE = store
    logger.info(f"Loaded {len(store)} API keys ({sum(1 for t in store.values() if t == Tier.ALPHA)} alpha, "
                f"{sum(1 for t in store.values() if t == Tier.PRO)} pro)")
    return store


def validate_api_key(key: str) -> Optional[Tier]:
    """Validate an API key and return its tier, or None if invalid."""
    store = _load_api_keys()
    return store.get(key)


def reload_api_keys() -> int:
    """Force reload API keys from environment (useful for hot-reload)."""
    global _API_KEY_STORE
    _API_KEY_STORE = None
    return len(_load_api_keys())


# ---------------------------------------------------------------------------
# Abuse detection — tracks failed requests per IP
# ---------------------------------------------------------------------------

@dataclass
class AbuseTracker:
    """Tracks suspicious behaviour per IP."""
    failed_requests: int = 0
    first_failure: float = field(default_factory=time.time)
    blocked_until: float = 0.0

    FAILURE_THRESHOLD = 10      # failures in window → block
    FAILURE_WINDOW = 300        # 5 minute window
    BLOCK_DURATION = 600        # block for 10 minutes

    def record_failure(self) -> bool:
        """Record a failure, return True if IP should now be blocked."""
        now = time.time()
        # Reset window if expired
        if now - self.first_failure > self.FAILURE_WINDOW:
            self.failed_requests = 0
            self.first_failure = now

        self.failed_requests += 1
        if self.failed_requests >= self.FAILURE_THRESHOLD:
            self.blocked_until = now + self.BLOCK_DURATION
            logger.warning(f"IP blocked due to {self.failed_requests} failures in {self.FAILURE_WINDOW}s")
            return True
        return False

    def is_blocked(self) -> bool:
        now = time.time()
        if self.blocked_until > now:
            return True
        if self.blocked_until > 0 and self.blocked_until <= now:
            # Unblock and reset
            self.blocked_until = 0.0
            self.failed_requests = 0
        return False


# ---------------------------------------------------------------------------
# reCAPTCHA v3 verification
# ---------------------------------------------------------------------------

RECAPTCHA_SECRET = os.getenv("SHRUTI_RECAPTCHA_SECRET", "")
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"
RECAPTCHA_MIN_SCORE = float(os.getenv("SHRUTI_RECAPTCHA_MIN_SCORE", "0.5"))

# Endpoints that require reCAPTCHA for anonymous users
RECAPTCHA_PROTECTED_ENDPOINTS = {"/transcribe", "/extract", "/ingest"}


async def verify_recaptcha(token: str) -> Tuple[bool, float]:
    """Verify a reCAPTCHA v3 token. Returns (valid, score)."""
    if not RECAPTCHA_SECRET:
        # reCAPTCHA not configured — skip verification
        logger.debug("reCAPTCHA not configured, skipping verification")
        return True, 1.0

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(RECAPTCHA_VERIFY_URL, data={
                "secret": RECAPTCHA_SECRET,
                "response": token,
            })
            result = resp.json()

        success = result.get("success", False)
        score = result.get("score", 0.0)

        if not success:
            logger.warning(f"reCAPTCHA verification failed: {result.get('error-codes', [])}")
            return False, 0.0

        if score < RECAPTCHA_MIN_SCORE:
            logger.warning(f"reCAPTCHA score too low: {score} < {RECAPTCHA_MIN_SCORE}")
            return False, score

        return True, score

    except Exception as e:
        logger.error(f"reCAPTCHA verification error: {e}")
        # Fail open on network errors to not break the service
        return True, 0.5


# ---------------------------------------------------------------------------
# Allowed CORS origins
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://avyay.ai",
    "https://www.avyay.ai",
    "https://shruti.avyay.ai",
    "https://api.avyay.ai",
]

# Allow localhost in development
if os.getenv("SHRUTI_ENV", "production").lower() in ("development", "dev", "local"):
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ])


# ---------------------------------------------------------------------------
# Rate Limiting Middleware
# ---------------------------------------------------------------------------

# Endpoints exempt from rate limiting
EXEMPT_ENDPOINTS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/config"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP + per-API-key rate limiting with tier-based limits,
    abuse detection, and reCAPTCHA enforcement.
    """

    def __init__(self, app):
        super().__init__(app)
        self._buckets: Dict[str, Bucket] = {}
        self._abuse: Dict[str, AbuseTracker] = defaultdict(AbuseTracker)
        self._cleanup_interval = 300  # cleanup stale entries every 5 min
        self._last_cleanup = time.time()

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP (supports proxies)."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        return request.client.host if request.client else "unknown"

    def _get_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request (header or query param)."""
        # Check Authorization header: Bearer <key> or ApiKey <key>
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        if auth.lower().startswith("apikey "):
            return auth[7:].strip()

        # Check X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key.strip()

        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key.strip()

        return None

    def _get_bucket_key(self, ip: str, api_key: Optional[str]) -> str:
        """Generate a unique bucket key."""
        if api_key:
            # Hash the API key for privacy in logs
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
            return f"key:{key_hash}"
        return f"ip:{ip}"

    def _cleanup_stale(self):
        """Periodically remove stale bucket entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        stale_keys = []
        for key, bucket in self._buckets.items():
            if now - bucket.last_refill > TIER_WINDOW * 2:
                stale_keys.append(key)

        for key in stale_keys:
            del self._buckets[key]

        # Clean abuse trackers
        stale_abuse = [ip for ip, tracker in self._abuse.items()
                       if not tracker.is_blocked() and tracker.failed_requests == 0]
        for ip in stale_abuse:
            del self._abuse[ip]

        self._last_cleanup = now
        if stale_keys or stale_abuse:
            logger.debug(f"Cleaned {len(stale_keys)} rate limit buckets, {len(stale_abuse)} abuse trackers")

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip exempt endpoints
        if path in EXEMPT_ENDPOINTS:
            return await call_next(request)

        # Skip OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        ip = self._get_client_ip(request)

        # --- Abuse check ---
        tracker = self._abuse[ip]
        if tracker.is_blocked():
            remaining = int(tracker.blocked_until - time.time())
            logger.warning(f"Blocked IP {ip} attempted access (unblocks in {remaining}s)")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Temporarily blocked due to suspicious activity",
                    "retry_after": remaining,
                },
                headers={"Retry-After": str(remaining)},
            )

        # --- API key & tier resolution ---
        api_key = self._get_api_key(request)
        tier = Tier.ANONYMOUS

        if api_key:
            resolved_tier = validate_api_key(api_key)
            if resolved_tier is None:
                # Invalid API key — record as failure
                tracker.record_failure()
                logger.warning(f"Invalid API key from {ip}")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid API key"},
                )
            tier = resolved_tier

        # --- reCAPTCHA for anonymous users on protected endpoints ---
        if tier == Tier.ANONYMOUS and path in RECAPTCHA_PROTECTED_ENDPOINTS:
            captcha_token = (
                request.headers.get("x-recaptcha-token")
                or request.query_params.get("recaptcha_token")
            )
            if RECAPTCHA_SECRET and not captcha_token:
                return JSONResponse(
                    status_code=403,
                    content={"error": "reCAPTCHA token required for anonymous access"},
                )
            if captcha_token:
                valid, score = await verify_recaptcha(captcha_token)
                if not valid:
                    tracker.record_failure()
                    return JSONResponse(
                        status_code=403,
                        content={"error": "reCAPTCHA verification failed", "score": score},
                    )

        # --- Rate limiting ---
        bucket_key = self._get_bucket_key(ip, api_key)
        limit = TIER_LIMITS[tier]

        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = Bucket(tokens=float(limit), max_tokens=limit)

        bucket = self._buckets[bucket_key]

        # If tier changed (e.g., key upgraded), update max tokens
        if bucket.max_tokens != limit:
            bucket.max_tokens = limit

        if not bucket.try_consume():
            retry_after = bucket.retry_after
            logger.info(f"Rate limit hit for {bucket_key} (tier={tier.value}, limit={limit}/h)")
            tracker.record_failure()
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "tier": tier.value,
                    "limit": f"{limit} requests/hour",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        # --- Periodic cleanup ---
        self._cleanup_stale()

        # --- Process request ---
        response = await call_next(request)

        # Add rate limit headers to successful responses
        remaining = max(0, int(bucket.tokens))
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Tier"] = tier.value

        # Record failures for non-2xx responses
        if response.status_code >= 400:
            tracker.record_failure()

        return response
