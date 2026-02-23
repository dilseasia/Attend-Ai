"""
dependencies.py
───────────────
FastAPI Depends() guards for JWT-protected routes.
Uses HTTPBearer so Swagger UI 🔒 Authorize button works automatically.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from jose import JWTError

from jwt_auth import verify_access_token, TokenPayload

# ── Type alias ────────────────────────────────────────────────────
CurrentUser = TokenPayload

# auto_error=False → we handle the missing token error ourselves
_bearer_scheme = HTTPBearer(auto_error=False)


def _get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> TokenPayload:
    """
    Base dependency — reads Bearer token and returns decoded JWT payload.
    Works with:
      • Swagger UI 🔒 Authorize button  (paste token only, no 'Bearer' prefix)
      • Postman  →  Authorization tab → Bearer Token
      • Any client sending:  Authorization: Bearer eyJ...
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Send: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials   # raw token, 'Bearer' already stripped by FastAPI
    try:
        return verify_access_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Role guards ───────────────────────────────────────────────────

def require_admin(user: TokenPayload = Depends(_get_current_user)) -> TokenPayload:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


def require_employee(user: TokenPayload = Depends(_get_current_user)) -> TokenPayload:
    if user.role != "employee":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Employee access required")
    return user


def require_any_role(user: TokenPayload = Depends(_get_current_user)) -> TokenPayload:
    return user


# ── Subdomain guards ──────────────────────────────────────────────

def require_same_subdomain(
    subdomain: str,
    user: TokenPayload = Depends(_get_current_user),
) -> TokenPayload:
    if user.subdomain != subdomain:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not belong to this tenant")
    return user


def require_admin_same_subdomain(
    subdomain: str,
    user: TokenPayload = Depends(require_admin),
) -> TokenPayload:
    if user.subdomain != subdomain:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not belong to this tenant")
    return user


def require_employee_same_subdomain(
    subdomain: str,
    user: TokenPayload = Depends(require_employee),
) -> TokenPayload:
    if user.subdomain != subdomain:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not belong to this tenant")
    return user