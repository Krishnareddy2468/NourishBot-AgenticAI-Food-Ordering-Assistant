"""Website authentication: password accounts, staff login, and Google OAuth."""

import asyncio
import base64
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import bcrypt
import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.auth.deps import extract_token
from app.auth.jwt import JWT_ALGORITHM, create_access_token
from app.core.config import settings
from app.db.models import Channel, Customer, OAuthIdentity, Restaurant, User
from app.db.session import AsyncSessionLocal

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
CUSTOMER_ROLE = "CUSTOMER"
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    return email


def validate_password(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(value) < 10:
        raise HTTPException(status_code=422, detail="Password must be at least 10 characters")
    if len(encoded) > 72:
        raise HTTPException(status_code=422, detail="Password is too long")
    return value


def verify_password(plain: str, hashed: str) -> bool:
    """Verify bcrypt, retaining local-only compatibility with legacy demo seeds."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return not settings.is_production and hmac.compare_digest(plain, hashed)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _set_session_cookie(response: Response, user: User) -> None:
    token = create_access_token(
        user_id=user.id,
        restaurant_id=user.restaurant_id,
        role=user.role,
        email=user.email,
        name=user.name,
    )
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=settings.JWT_EXPIRY_MINUTES * 60,
        httponly=True,
        secure=settings.is_production,
        samesite=settings.AUTH_COOKIE_SAMESITE,
        path="/",
    )


def _login_response(user: User) -> "LoginResponse":
    return LoginResponse(
        user_id=user.id,
        name=user.name,
        email=user.email or "",
        role=user.role,
        restaurant_id=user.restaurant_id,
    )


async def _ensure_registration_restaurant(session) -> bool:
    restaurant_exists = await session.scalar(
        select(Restaurant.id).where(Restaurant.id == settings.DEFAULT_RESTAURANT_ID)
    )
    if restaurant_exists:
        return True
    if settings.is_production:
        return False

    session.add(
        Restaurant(
            id=settings.DEFAULT_RESTAURANT_ID,
            name=settings.RESTAURANT_NAME,
            address=settings.RESTAURANT_LOCATION,
            timezone="Asia/Kolkata",
        )
    )
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return bool(
            await session.scalar(
                select(Restaurant.id).where(Restaurant.id == settings.DEFAULT_RESTAURANT_ID)
            )
        )
    await session.execute(
        text(
            "SELECT setval(pg_get_serial_sequence('restaurants', 'id'), "
            "GREATEST((SELECT COALESCE(MAX(id), 1) FROM restaurants), 1))"
        )
    )
    return True


async def _provision_customer_profile(session, user: User) -> None:
    """Bind a website CUSTOMER account to the domain customer/channel identity."""
    if user.role != CUSTOMER_ROLE:
        return
    if user.id is None:
        await session.flush()
    customer = None
    if user.customer_id:
        customer = await session.get(Customer, user.customer_id)
    if customer is None and user.email:
        customer = await session.scalar(select(Customer).where(
            Customer.restaurant_id == user.restaurant_id,
            func.lower(Customer.email) == user.email.lower(),
        ).order_by(Customer.id).limit(1))
    if customer is None:
        customer = Customer(
            restaurant_id=user.restaurant_id,
            name=user.name,
            phone=f"web-account-{user.id}",
            email=user.email,
        )
        session.add(customer)
        await session.flush()
    user.customer_id = customer.id
    external_id = f"account:{user.id}"
    channel = await session.scalar(select(Channel).where(
        Channel.restaurant_id == user.restaurant_id,
        Channel.type == "WEB",
        Channel.external_id == external_id,
    ))
    if channel is None:
        now = datetime.now(timezone.utc)
        session.add(Channel(
            restaurant_id=user.restaurant_id,
            type="WEB",
            external_id=external_id,
            customer_id=customer.id,
            verified_at=now,
            verification_method="AUTHENTICATED_WEB_ACCOUNT",
            linked_at=now,
        ))
    elif channel.customer_id != customer.id:
        raise HTTPException(status_code=409, detail="Web account identity is already linked")


def create_oauth_state() -> tuple[str, str, str, str]:
    """Return signed cookie value, state, nonce, and PKCE verifier."""
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    payload = {
        "typ": "oauth_state",
        "state": state,
        "nonce": nonce,
        "verifier": verifier,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.OAUTH_STATE_TTL_SECONDS),
    }
    signed = jwt.encode(payload, settings.JWT_SECRET, algorithm=JWT_ALGORITHM)
    return signed, state, nonce, verifier


def decode_oauth_state(value: str) -> dict:
    payload = jwt.decode(value, settings.JWT_SECRET, algorithms=[JWT_ALGORITHM])
    if payload.get("typ") != "oauth_state":
        raise jwt.InvalidTokenError("Invalid OAuth state type")
    return payload


def build_google_authorization_url(state: str, nonce: str, verifier: str) -> str:
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{urlencode(params)}"


def _frontend_redirect(path: str, **query: str) -> RedirectResponse:
    suffix = f"?{urlencode(query)}" if query else ""
    return RedirectResponse(f"{settings.FRONTEND_URL}{path}{suffix}", status_code=303)


def _oauth_error(message: str) -> RedirectResponse:
    response = _frontend_redirect("/login", oauth_error=message)
    response.delete_cookie(
        settings.OAUTH_STATE_COOKIE_NAME,
        path="/api/v1/auth/oauth/google",
        secure=settings.is_production,
        httponly=True,
        samesite="lax",
    )
    return response


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=256)


class LoginResponse(BaseModel):
    user_id: int
    name: str = ""
    email: str
    role: str
    restaurant_id: int


class OAuthProvidersResponse(BaseModel):
    google: bool


@router.get("/oauth/providers", response_model=OAuthProvidersResponse)
async def oauth_providers():
    return OAuthProvidersResponse(google=settings.google_oauth_enabled)


@router.post("/signup", response_model=LoginResponse, status_code=201)
async def signup(body: SignupRequest, response: Response):
    name = body.name.strip()
    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Enter your name")
    email = normalize_email(body.email)
    password = validate_password(body.password)

    async with AsyncSessionLocal() as session:
        if not await _ensure_registration_restaurant(session):
            raise HTTPException(status_code=503, detail="Account registration is not ready")

        existing = await session.scalar(select(User.id).where(func.lower(User.email) == email))
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        user = User(
            restaurant_id=settings.DEFAULT_RESTAURANT_ID,
            name=name,
            email=email,
            password_hash=hash_password(password),
            role=CUSTOMER_ROLE,
            active=True,
            email_verified=False,
        )
        session.add(user)
        try:
            await session.flush()
            await _provision_customer_profile(session, user)
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        await session.refresh(user)

    _set_session_cookie(response, user)
    return _login_response(user)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response):
    email = normalize_email(body.email)
    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User).where(func.lower(User.email) == email, User.active.is_(True))
        )

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    _set_session_cookie(response, user)
    return _login_response(user)


@router.get("/oauth/google/start")
async def google_oauth_start():
    if not settings.google_oauth_enabled:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")

    signed, state, nonce, verifier = create_oauth_state()
    response = RedirectResponse(build_google_authorization_url(state, nonce, verifier), status_code=302)
    response.set_cookie(
        key=settings.OAUTH_STATE_COOKIE_NAME,
        value=signed,
        max_age=settings.OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        path="/api/v1/auth/oauth/google",
    )
    return response


@router.get("/oauth/google/callback")
async def google_oauth_callback(request: Request):
    if not settings.google_oauth_enabled:
        return _oauth_error("Google sign-in is not configured")
    if request.query_params.get("error"):
        return _oauth_error("Google sign-in was cancelled")

    code = request.query_params.get("code", "")
    returned_state = request.query_params.get("state", "")
    signed_state = request.cookies.get(settings.OAUTH_STATE_COOKIE_NAME, "")
    if not code or not returned_state or not signed_state:
        return _oauth_error("Google sign-in could not be verified")

    try:
        state_data = decode_oauth_state(signed_state)
    except Exception:
        return _oauth_error("Google sign-in expired. Please try again")
    if not hmac.compare_digest(returned_state, str(state_data.get("state", ""))):
        return _oauth_error("Google sign-in could not be verified")

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            token_response = await client.post(
                GOOGLE_TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                    "code_verifier": state_data["verifier"],
                },
            )
            token_response.raise_for_status()
            token_data = token_response.json()
        raw_id_token = token_data.get("id_token")
        if not raw_id_token:
            return _oauth_error("Google did not return an identity token")
        claims = await asyncio.to_thread(
            google_id_token.verify_oauth2_token,
            raw_id_token,
            GoogleRequest(),
            settings.GOOGLE_CLIENT_ID,
        )
    except (httpx.HTTPError, ValueError, KeyError):
        return _oauth_error("Google sign-in failed. Please try again")

    if not hmac.compare_digest(str(claims.get("nonce", "")), str(state_data.get("nonce", ""))):
        return _oauth_error("Google sign-in could not be verified")
    if claims.get("email_verified") is not True:
        return _oauth_error("Google must verify your email before sign-in")

    provider_subject = str(claims.get("sub", ""))
    try:
        email = normalize_email(str(claims.get("email", "")))
    except HTTPException:
        return _oauth_error("Google account email is incomplete")
    name = str(claims.get("name") or email.split("@", 1)[0]).strip()[:80]
    if not provider_subject:
        return _oauth_error("Google account identity is incomplete")

    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User)
            .join(OAuthIdentity, OAuthIdentity.user_id == User.id)
            .where(
                OAuthIdentity.provider == "google",
                OAuthIdentity.provider_subject == provider_subject,
            )
        )
        if user is None:
            user = await session.scalar(select(User).where(func.lower(User.email) == email))
            if user is None:
                if not await _ensure_registration_restaurant(session):
                    return _oauth_error("Account registration is not ready")
                user = User(
                    restaurant_id=settings.DEFAULT_RESTAURANT_ID,
                    name=name,
                    email=email,
                    password_hash=hash_password(secrets.token_urlsafe(48)),
                    role=CUSTOMER_ROLE,
                    active=True,
                    email_verified=True,
                )
                session.add(user)
                await session.flush()
            else:
                user.email_verified = True
                if not user.name:
                    user.name = name
            session.add(
                OAuthIdentity(
                    user_id=user.id,
                    provider="google",
                    provider_subject=provider_subject,
                    email=email,
                )
            )
            await _provision_customer_profile(session, user)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return _oauth_error("This Google account is already linked")
            await session.refresh(user)
        elif user.role == CUSTOMER_ROLE and not user.customer_id:
            await _provision_customer_profile(session, user)
            await session.commit()
            await session.refresh(user)

    if not user.active:
        return _oauth_error("This account is disabled")

    response = _frontend_redirect("/workspace")
    response.delete_cookie(
        settings.OAUTH_STATE_COOKIE_NAME,
        path="/api/v1/auth/oauth/google",
        secure=settings.is_production,
        httponly=True,
        samesite="lax",
    )
    _set_session_cookie(response, user)
    return response


@router.get("/me", response_model=LoginResponse)
async def me(user=Depends(extract_token)):
    return LoginResponse(
        user_id=int(user["user_id"]),
        name=str(user.get("name") or ""),
        email=str(user.get("email") or ""),
        role=str(user["role"]),
        restaurant_id=int(user["restaurant_id"]),
    )


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.is_production,
        httponly=True,
        samesite=settings.AUTH_COOKIE_SAMESITE,
    )
