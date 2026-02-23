"""
main.py  (JWT-enabled version)
──────────────────────────────
Changes from original:
  • /api/{subdomain}/login        → now returns JWT access + refresh tokens
  • /api/{subdomain}/token/refresh→ new: exchange refresh token for new access token
  • /api/{subdomain}/logout       → new: client-side logout helper
  • All employee write routes     → protected with JWT Depends()
  • Admin-only routes             → protected with require_admin_same_subdomain()
  • Cross-tenant protection       → token subdomain must match URL subdomain
"""
import json
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import re
from datetime import datetime, date

from jose import JWTError
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form


from config import PLAN_LIMITS, POSTGRES_ADMIN_CONFIG
from pydantic_models import (
    CompanyRegisterRequest, SubdomainCheckRequest, PlanUpgradeRequest,
    EmployeeRegisterRequest, EmployeeLoginRequest, ChangePasswordRequest, EmployeeUpdateRequest,
    EncodingStatusPatchRequest
)
from helper_functions import (
    get_master_connection, get_tenant_conn,
    create_database_if_not_exists, provision_tenant_database,
    register_company_in_master, create_admin_in_tenant,
    hash_password, verify_password, check_employee_limit, generate_employee_id,cosine_similarity,align_face_improved,_compute_sample_score,_get_emp_internal_id,_decode_upload
)

# ── JWT imports ───────────────────────────────────────────────────
from jwt_auth import (
    create_token_pair, verify_access_token, verify_refresh_token,
    create_access_token, ACCESS_TOKEN_EXPIRE_MIN,
)
from dependencies import (
    require_admin, require_employee, require_any_role,
    require_same_subdomain, require_admin_same_subdomain,
    require_employee_same_subdomain, CurrentUser,
)
face_app = None


app = FastAPI(
    title="AttendAI — Company Provisioning API",
    swagger_ui_init_oauth={},
    openapi_tags=[],
)

# Enables the 🔒 Authorize button in Swagger UI
bearer_scheme = HTTPBearer(auto_error=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


# ─────────────────────────────────────────────────────────────────
#  AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────

# ── UNIFIED LOGIN → returns JWT pair ─────────────────────────────
@app.post("/api/{subdomain}/login")
def unified_login(subdomain: str, payload: EmployeeLoginRequest):
    """
    Login for both admins and employees.
    Returns: access_token, refresh_token, role, user info.

    POST /api/acme/login
    Body: { "email": "...", "password": "..." }

    Authorization flow after login:
        1. Store access_token in memory (short-lived, 60 min)
        2. Store refresh_token in httpOnly cookie or secure storage (7 days)
        3. Send "Authorization: Bearer <access_token>" on every protected request
        4. When access_token expires, call POST /api/{subdomain}/token/refresh
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # ── Try admin first ──────────────────────────────────────
        cur.execute(
            "SELECT id, email, password, full_name, phone, is_active, last_login, created_at "
            "FROM admins WHERE email = %s",
            (payload.email,),
        )
        admin = cur.fetchone()

        if admin:
            if not verify_password(payload.password, admin["password"]):
                raise HTTPException(status_code=401, detail="Invalid email or password")
            if not admin["is_active"]:
                raise HTTPException(status_code=403, detail="Admin account is deactivated.")

            cur.execute("UPDATE admins SET last_login = NOW() WHERE id = %s", (admin["id"],))
            conn.commit()
            cur.close()

            tokens = create_token_pair(
                user_id=admin["id"], subdomain=subdomain, role="admin"
            )
            logging.info(f"[+] Admin '{payload.email}' logged in to '{subdomain}'")
            return {
                "success": True,
                "message": "Login successful",
                **tokens,
                "user": {
                    "id":         admin["id"],
                    "email":      admin["email"],
                    "full_name":  admin["full_name"],
                    "phone":      admin["phone"],
                    "is_active":  admin["is_active"],
                    "last_login": str(admin["last_login"]) if admin["last_login"] else None,
                    "created_at": str(admin["created_at"]),
                },
            }

        # ── Try employee ─────────────────────────────────────────
        cur.execute(
            "SELECT id, employee_id, full_name, email, password, department, created_at, expire_at "
            "FROM employees WHERE email = %s",
            (payload.email,),
        )
        emp = cur.fetchone()

        if not emp:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not verify_password(payload.password, emp["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if emp["expire_at"] and emp["expire_at"].date() < date.today():
            raise HTTPException(status_code=403, detail="Your account has expired. Contact your administrator.")

        cur.close()

        tokens = create_token_pair(
            user_id=emp["id"], subdomain=subdomain, role="employee"
        )
        logging.info(f"[+] Employee '{payload.email}' logged in to '{subdomain}'")
        return {
            "success": True,
            "message": "Login successful",
            **tokens,
            "user": {
                "id":          emp["id"],
                "employee_id": emp["employee_id"],
                "full_name":   emp["full_name"],
                "email":       emp["email"],
                "department":  emp["department"],
                "created_at":  str(emp["created_at"]),
                "expire_at":   str(emp["expire_at"]) if emp["expire_at"] else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Unified login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ── REFRESH TOKEN → new access token ─────────────────────────────
class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/api/{subdomain}/token/refresh")
def refresh_access_token(subdomain: str, payload: RefreshRequest):
    """
    Exchange a valid refresh token for a new access token.

    POST /api/acme/token/refresh
    Body: { "refresh_token": "<token>" }
    """
    try:
        data = verify_refresh_token(payload.refresh_token)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired refresh token: {e}")

    if data.subdomain != subdomain:
        raise HTTPException(status_code=403, detail="Refresh token does not belong to this tenant")

    new_access = create_access_token(
        user_id=data.user_id, subdomain=data.subdomain, role=data.role
    )
    return {
        "access_token": new_access,
        "token_type":   "bearer",
        "expires_in":   ACCESS_TOKEN_EXPIRE_MIN * 60,
        "role":         data.role,
        "subdomain":    data.subdomain,
    }


# ── LOGOUT (client-side; informational) ──────────────────────────
@app.post("/api/{subdomain}/logout")
def logout(subdomain: str, _: CurrentUser = Depends(require_same_subdomain)):
    """
    Logout endpoint.
    JWTs are stateless — actual invalidation is done client-side by
    discarding both tokens. This endpoint just confirms the action.

    POST /api/acme/logout
    Header: Authorization: Bearer <access_token>

    NOTE: For true server-side revocation, store a token blacklist in Redis
    and check it inside verify_access_token().
    """
    return {"success": True, "message": "Logged out. Please discard your tokens."}


# ── GET CURRENT USER (me) ─────────────────────────────────────────
@app.get("/api/{subdomain}/me")
def get_me(subdomain: str, user: CurrentUser = Depends(require_same_subdomain)):
    """
    Returns the decoded JWT payload of the currently authenticated user.

    GET /api/acme/me
    Header: Authorization: Bearer <access_token>
    """
    return {
        "success":   True,
        "user_id":   user.user_id,
        "role":      user.role,
        "subdomain": user.subdomain,
    }


# ─────────────────────────────────────────────────────────────────
#  COMPANY ENDPOINTS  (unchanged, no auth needed for provisioning)
# ─────────────────────────────────────────────────────────────────

@app.get("/api/company/check-subdomain")
def check_subdomain(subdomain: str = Query(..., min_length=2, max_length=100)):
    subdomain = subdomain.lower().strip()
    if not re.match(r'^[a-z0-9][a-z0-9\-]*$', subdomain):
        raise HTTPException(status_code=400, detail="Subdomain must be lowercase letters, numbers, and hyphens only")
    try:
        conn = get_master_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM companies WHERE subdomain = %s", (subdomain,))
        exists = cur.fetchone()
        cur.close(); conn.close()
        if exists:
            return {"available": False, "subdomain": subdomain,
                    "message": f"'{subdomain}' is already taken",
                    "suggestions": [f"{subdomain}1", f"{subdomain}-hq", f"{subdomain}-india", f"my{subdomain}"]}
        return {"available": True, "subdomain": subdomain, "message": f"'{subdomain}' is available! 🎉"}
    except Exception as e:
        logging.error(f"❌ Subdomain check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check subdomain availability")


@app.post("/api/company/register")
def register_company(payload: CompanyRegisterRequest):
    subdomain = payload.subdomain.lower()
    plan_info = PLAN_LIMITS[payload.plan]
    try:
        conn = get_master_connection(); cur = conn.cursor()
        cur.execute("SELECT id FROM companies WHERE subdomain = %s", (subdomain,))
        if cur.fetchone():
            cur.close(); conn.close()
            raise HTTPException(status_code=409, detail=f"Subdomain '{subdomain}' is already registered.")
        cur.close(); conn.close()
        db_name    = provision_tenant_database(subdomain)
        company_id = register_company_in_master(payload.company_name, subdomain, payload.plan, plan_info["max_employees"], db_name)
        admin_id   = create_admin_in_tenant(subdomain, payload.admin_full_name, payload.admin_email, payload.admin_password, payload.admin_phone)
        return {
            "success": True,
            "message": "Company registered successfully! 🎉",
            "company": {"id": company_id, "name": payload.company_name, "subdomain": subdomain,
                        "plan": payload.plan, "max_employees": plan_info["max_employees"],
                        "dashboard_url": f"https://attendai.app/{subdomain}"},
            "admin": {"id": admin_id, "email": payload.admin_email},
        }
    except HTTPException: raise
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Company or admin email already exists")
    except Exception as e:
        logging.error(f"❌ Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/company/list")
def list_companies(plan: Optional[str] = None, active_only: bool = True, limit: int = 50, offset: int = 0):
    try:
        conn = get_master_connection(); cur = conn.cursor(cursor_factory=RealDictCursor)
        query = "SELECT * FROM companies WHERE 1=1"; params = []
        if active_only: query += " AND is_active_subscription = TRUE"
        if plan:
            if plan not in PLAN_LIMITS: raise HTTPException(status_code=400, detail=f"Invalid plan: {plan}")
            query += " AND plan = %s"; params.append(plan)
        cur.execute(f"SELECT COUNT(*) FROM ({query}) AS t", params)
        total = cur.fetchone()["count"]
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"; params.extend([limit, offset])
        cur.execute(query, params); companies = cur.fetchall(); cur.close(); conn.close()
        return {"success": True, "total": total, "companies": [
            {"id": c["id"], "name": c["name"], "subdomain": c["subdomain"], "plan": c["plan"],
             "is_active_subscription": c["is_active_subscription"], "created_at": str(c["created_at"])}
            for c in companies]}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/company/{subdomain}")
def get_company(subdomain: str):
    try:
        conn = get_master_connection(); cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT c.*, cd.db_name, cd.db_host, cd.is_provisioned, cd.provisioned_at
            FROM companies c LEFT JOIN company_databases cd ON c.id = cd.company_id
            WHERE c.subdomain = %s""", (subdomain.lower(),))
        company = cur.fetchone(); cur.close(); conn.close()
        if not company: raise HTTPException(status_code=404, detail=f"Company '{subdomain}' not found")
        return {"success": True, "company": {**{k: str(v) if isinstance(v, datetime) else v for k, v in company.items()}}}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
#  EMPLOYEE ENDPOINTS  (JWT-protected)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/{subdomain}/employees/register", status_code=201)
def register_employee(
    subdomain: str,
    payload: EmployeeRegisterRequest,
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    """
    Register a new employee.  Admin JWT required.

    POST /api/acme/employees/register
    Header: Authorization: Bearer <admin_access_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        check_employee_limit(subdomain, conn)
        emp_id = payload.employee_id or generate_employee_id(subdomain, conn)
        cur = conn.cursor()
        cur.execute("SELECT id FROM employees WHERE email = %s", (payload.email,))
        if cur.fetchone(): raise HTTPException(status_code=409, detail=f"Email '{payload.email}' already registered")
        cur.execute("SELECT id FROM employees WHERE employee_id = %s", (emp_id,))
        if cur.fetchone(): raise HTTPException(status_code=409, detail=f"Employee ID '{emp_id}' already exists")
        cur.execute("""
            INSERT INTO employees (employee_id, full_name, email, password, department, created_at, expire_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s)
            RETURNING id, employee_id, full_name, email, department, created_at, expire_at
        """, (emp_id, payload.full_name, payload.email, hash_password(payload.password), payload.department, payload.expire_at))
        row = cur.fetchone(); conn.commit(); cur.close()
        return {"success": True, "message": f"Employee '{payload.full_name}' registered",
                "employee": {"id": row[0], "employee_id": row[1], "full_name": row[2],
                             "email": row[3], "department": row[4],
                             "created_at": str(row[5]), "expire_at": str(row[6]) if row[6] else None}}
    except HTTPException: raise
    except psycopg2.errors.UniqueViolation:
        conn.rollback(); raise HTTPException(status_code=409, detail="Email or Employee ID already exists")
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/{subdomain}/employees")
def list_employees(
    subdomain:   str,
    department:  Optional[str] = Query(None),
    active_only: bool          = Query(True),
    search:      Optional[str] = Query(None),
    limit:       int           = Query(50, ge=1, le=200),
    offset:      int           = Query(0, ge=0),
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        conditions = ["1=1"]; params = []
        if active_only: conditions.append("(expire_at IS NULL OR expire_at >= CURRENT_DATE)")
        if department:  conditions.append("department ILIKE %s"); params.append(f"%{department}%")
        if search:
            conditions.append("(full_name ILIKE %s OR email ILIKE %s OR employee_id ILIKE %s)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        where = " AND ".join(conditions)
        cur.execute(f"SELECT COUNT(*) FROM employees WHERE {where}", params)
        total = cur.fetchone()["count"]
        cur.execute(f"""
            SELECT e.id, e.employee_id, e.full_name, e.email, e.department,
                   e.created_at, e.expire_at, COUNT(DISTINCT fe.id) AS face_encodings_count
            FROM employees e
            LEFT JOIN face_encodings fe ON fe.emp_id = e.id AND fe.is_active = TRUE
            WHERE {where} GROUP BY e.id ORDER BY e.created_at DESC LIMIT %s OFFSET %s
        """, params + [limit, offset])
        employees = cur.fetchall(); cur.close()
        return {"success": True, "total": total, "page": offset // limit + 1,
                "employees": [{"id": e["id"], "employee_id": e["employee_id"], "full_name": e["full_name"],
                               "email": e["email"], "department": e["department"],
                               "is_active": not (e["expire_at"] and e["expire_at"].date() < date.today()),
                               "face_registered": e["face_encodings_count"] > 0,
                               "created_at": str(e["created_at"]),
                               "expire_at": str(e["expire_at"]) if e["expire_at"] else None}
                              for e in employees]}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.get("/api/{subdomain}/employees/{employee_id}")
def get_employee(
    subdomain:   str,
    employee_id: str,
    user: CurrentUser = Depends(require_same_subdomain),         # ← ANY authenticated user
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT e.id, e.employee_id, e.full_name, e.email, e.department,
                   e.created_at, e.expire_at,
                   COUNT(DISTINCT al.id) AS total_attendance,
                   COUNT(DISTINCT fe.id) AS face_encodings_count
            FROM employees e
            LEFT JOIN attendance_logs al ON al.user_id = e.id
            LEFT JOIN face_encodings  fe ON fe.emp_id  = e.id AND fe.is_active = TRUE
            WHERE e.employee_id = %s GROUP BY e.id
        """, (employee_id,))
        emp = cur.fetchone(); cur.close()
        if not emp: raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        is_active = not (emp["expire_at"] and emp["expire_at"].date() < date.today())
        return {"success": True, "employee": {
            "id": emp["id"], "employee_id": emp["employee_id"], "full_name": emp["full_name"],
            "email": emp["email"], "department": emp["department"], "is_active": is_active,
            "face_registered": emp["face_encodings_count"] > 0,
            "total_attendance": emp["total_attendance"],
            "created_at": str(emp["created_at"]),
            "expire_at": str(emp["expire_at"]) if emp["expire_at"] else None}}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.patch("/api/{subdomain}/employees/{employee_id}")
def update_employee(
    subdomain:   str,
    employee_id: str,
    payload:     EmployeeUpdateRequest,
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM employees WHERE employee_id = %s", (employee_id,))
        if not cur.fetchone(): raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        updates = {}
        if payload.full_name  is not None: updates["full_name"]  = payload.full_name
        if payload.department is not None: updates["department"] = payload.department
        if payload.expire_at  is not None: updates["expire_at"]  = payload.expire_at
        if not updates: raise HTTPException(status_code=400, detail="No fields provided to update")
        set_clause = ", ".join([f"{k} = %s" for k in updates])
        values     = list(updates.values()) + [employee_id]
        cur.execute(f"UPDATE employees SET {set_clause} WHERE employee_id = %s "
                    f"RETURNING id, employee_id, full_name, email, department, expire_at", values)
        row = cur.fetchone(); conn.commit(); cur.close()
        return {"success": True, "message": "Employee updated",
                "employee": {"id": row[0], "employee_id": row[1], "full_name": row[2],
                             "email": row[3], "department": row[4],
                             "expire_at": str(row[5]) if row[5] else None}}
    except HTTPException: raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.patch("/api/{subdomain}/employees/{employee_id}/change-password")
def change_employee_password(
    subdomain:   str,
    employee_id: str,
    payload:     ChangePasswordRequest,
    user: CurrentUser = Depends(require_same_subdomain),         # ← SELF (any auth user)
):
    """Employee changes their own password. Token subdomain must match."""
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM employees WHERE employee_id = %s", (employee_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        if not verify_password(payload.current_password, row[1]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        cur.execute("UPDATE employees SET password = %s WHERE employee_id = %s",
                    (hash_password(payload.new_password), employee_id))
        conn.commit(); cur.close()
        return {"success": True, "message": "Password changed successfully"}
    except HTTPException: raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.patch("/api/{subdomain}/employees/{employee_id}/deactivate")
def deactivate_employee(
    subdomain:   str,
    employee_id: str,
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE employees SET expire_at = CURRENT_DATE WHERE employee_id = %s RETURNING id, full_name, email", (employee_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        conn.commit(); cur.close()
        return {"success": True, "message": f"Employee '{row[1]}' deactivated", "employee_id": employee_id}
    except HTTPException: raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.patch("/api/{subdomain}/employees/{employee_id}/reactivate")
def reactivate_employee(
    subdomain:   str,
    employee_id: str,
    expire_at:   Optional[str] = Query(None),
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE employees SET expire_at = %s WHERE employee_id = %s RETURNING id, full_name", (expire_at, employee_id))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        conn.commit(); cur.close()
        return {"success": True, "message": f"Employee '{row[1]}' reactivated", "employee_id": employee_id, "expire_at": expire_at}
    except HTTPException: raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


@app.delete("/api/{subdomain}/employees/{employee_id}")
def delete_employee(
    subdomain:   str,
    employee_id: str,
    user: CurrentUser = Depends(require_admin_same_subdomain),   # ← ADMIN ONLY
):
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM employees WHERE employee_id = %s RETURNING id, full_name", (employee_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
        conn.commit(); cur.close()
        return {"success": True, "message": f"Employee '{row[1]}' permanently deleted", "employee_id": employee_id}
    except HTTPException: raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


# ═════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
#  1. Register face encodings from images  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/{subdomain}/employees/{employee_id}/face-encodings", status_code=201)
async def register_face_encoding(
    subdomain:   str,
    employee_id: str,
    photos: list[UploadFile] = File(..., description="1 or more face images"),
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    Upload 1-N face images — InsightFace handles everything automatically.

    Per image the API:
      - Detects face with InsightFace buffalo_l
      - Tries face alignment (same as fastapi_server.py)
      - Extracts normed_embedding (512-d), det_score, sample_score
      - Inserts into face_encodings table

    Rules:
      ✅ Exactly 1 face detected → stored
      ❌ 0 faces detected        → reported in "failed" list, no crash
      ❌ >1 faces detected       → reported in "failed" list (ambiguous)

    All valid images are committed in a single transaction.

    POST /api/acme/employees/EMP001/face-encodings
    Header: Authorization: Bearer <admin_token>
    Form:   photos = <file1>, <file2>, ...

    Response:
    {
      "success": true,
      "message": "3 encoding(s) saved, 1 failed",
      "employee_id": "EMP001",
      "saved": [
        {
          "file": "front.jpg",
          "encoding_id": 12,
          "sample_score": 0.87,
          "confidence_score": 0.99,
          "is_active": true,
          "created_at": "2026-02-23 10:00:00"
        }
      ],
      "failed": [
        { "file": "blurry.jpg", "reason": "No face detected in image" }
      ],
      "total_active_encodings": 5
    }
    """
    if not photos:
        raise HTTPException(status_code=400, detail="At least one image is required")

    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        success_rows = []
        failed_files = []

        for upload in photos:
            filename = upload.filename or "unknown"

            # ── Step 1: Decode ────────────────────────────────────
            frame = await _decode_upload(upload)
            if frame is None:
                failed_files.append({
                    "file":   filename,
                    "reason": "Could not decode image (invalid or corrupt file)",
                })
                continue

            # ── Step 2: Detect faces ──────────────────────────────
            faces = face_app.get(frame)

            if len(faces) == 0:
                failed_files.append({
                    "file":   filename,
                    "reason": "No face detected in image",
                })
                continue

            if len(faces) > 1:
                failed_files.append({
                    "file":   filename,
                    "reason": f"{len(faces)} faces detected — upload a single-face image",
                })
                continue

            face = faces[0]

            # ── Step 3: Try alignment (same as fastapi_server.py) ─
            aligned = align_face_improved(frame, face)
            if aligned is not None:
                aligned_faces = face_app.get(aligned)
                if aligned_faces:
                    face = aligned_faces[0]

            # ── Step 4: Extract values ────────────────────────────
            embedding        = face.normed_embedding.tolist()        # 512-d
            confidence_score = round(float(face.det_score), 4)
            sample_score     = _compute_sample_score(face, frame.shape)

            # ── Step 5: Insert ────────────────────────────────────
            cur.execute("""
                INSERT INTO face_encodings
                    (emp_id, encoding_vector, sample_score, confidence_score,
                     is_active, created_at, updated_at)
                VALUES (%s, %s::jsonb, %s, %s, TRUE, NOW(), NOW())
                RETURNING id, emp_id, sample_score, confidence_score, is_active, created_at
            """, (
                emp_internal_id,
                json.dumps(embedding),
                sample_score,
                confidence_score,
            ))
            row = cur.fetchone()
            success_rows.append({
                "file":             filename,
                "encoding_id":      row["id"],
                "sample_score":     row["sample_score"],
                "confidence_score": row["confidence_score"],
                "is_active":        row["is_active"],
                "created_at":       str(row["created_at"]),
            })

        # ── Commit if at least one image succeeded ────────────────
        if success_rows:
            conn.commit()
        else:
            conn.rollback()

        # ── Count total active encodings for this employee ────────
        cur.execute(
            "SELECT COUNT(*) as cnt FROM face_encodings WHERE emp_id = %s AND is_active = TRUE",
            (emp_internal_id,),
        )
        total_active = cur.fetchone()["cnt"]
        cur.close()

        logging.info(
            f"[+] Face encodings for '{employee_id}' @ '{subdomain}' "
            f"— saved: {len(success_rows)}, failed: {len(failed_files)}"
        )

        return {
            "success":                len(success_rows) > 0,
            "message":                f"{len(success_rows)} encoding(s) saved, {len(failed_files)} failed",
            "employee_id":            employee_id,
            "saved":                  success_rows,
            "failed":                 failed_files,
            "total_active_encodings": total_active,
        }

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logging.error(f"❌ register_face_encoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  2. list encodings for an employee  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.get("/api/{subdomain}/employees/{employee_id}/face-encodings")
def list_face_encodings(
    subdomain:      str,
    employee_id:    str,
    active_only:    bool = Query(True,  description="Only return is_active=TRUE encodings"),
    include_vector: bool = Query(False, description="Include raw 512-d vector (heavy payload)"),
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    GET /api/acme/employees/EMP001/face-encodings
    GET /api/acme/employees/EMP001/face-encodings?active_only=false&include_vector=true
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        vector_col = ", encoding_vector" if include_vector else ""
        condition  = "AND is_active = TRUE" if active_only else ""

        cur.execute(f"""
            SELECT id, emp_id {vector_col}, sample_score, confidence_score,
                   is_active, created_at, updated_at
            FROM face_encodings
            WHERE emp_id = %s {condition}
            ORDER BY created_at DESC
        """, (emp_internal_id,))

        rows = cur.fetchall()
        cur.close()

        return {
            "success":        True,
            "employee_id":    employee_id,
            "total":          len(rows),
            "face_encodings": [
                {
                    "id":               r["id"],
                    "sample_score":     r["sample_score"],
                    "confidence_score": r["confidence_score"],
                    "is_active":        r["is_active"],
                    "created_at":       str(r["created_at"]),
                    "updated_at":       str(r["updated_at"]),
                    **({"encoding_vector": r["encoding_vector"]} if include_vector else {}),
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  3. Get single encoding  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.get("/api/{subdomain}/employees/{employee_id}/face-encodings/{enc_id}")
def get_face_encoding(
    subdomain:      str,
    employee_id:    str,
    enc_id:         int,
    include_vector: bool = Query(False),
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    GET /api/acme/employees/EMP001/face-encodings/3
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        vector_col = ", encoding_vector" if include_vector else ""
        cur.execute(f"""
            SELECT id, emp_id {vector_col}, sample_score, confidence_score,
                   is_active, created_at, updated_at
            FROM face_encodings
            WHERE id = %s AND emp_id = %s
        """, (enc_id, emp_internal_id))

        row = cur.fetchone()
        cur.close()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Face encoding #{enc_id} not found for employee '{employee_id}'"
            )

        result = {
            "id":               row["id"],
            "emp_id":           row["emp_id"],
            "sample_score":     row["sample_score"],
            "confidence_score": row["confidence_score"],
            "is_active":        row["is_active"],
            "created_at":       str(row["created_at"]),
            "updated_at":       str(row["updated_at"]),
        }
        if include_vector:
            result["encoding_vector"] = row["encoding_vector"]

        return {"success": True, "encoding": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  4. Activate / Deactivate an encoding  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.patch("/api/{subdomain}/employees/{employee_id}/face-encodings/{enc_id}")
def toggle_face_encoding_status(
    subdomain:   str,
    employee_id: str,
    enc_id:      int,
    payload:     EncodingStatusPatchRequest,
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    Deactivate a low-quality sample without deleting it.

    PATCH /api/acme/employees/EMP001/face-encodings/3
    Body:   { "is_active": false }
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        cur.execute("""
            UPDATE face_encodings
            SET is_active = %s, updated_at = NOW()
            WHERE id = %s AND emp_id = %s
            RETURNING id, is_active, updated_at
        """, (payload.is_active, enc_id, emp_internal_id))

        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Face encoding #{enc_id} not found for employee '{employee_id}'"
            )

        conn.commit()
        cur.close()

        return {
            "success":    True,
            "message":    f"Face encoding #{enc_id} {'activated' if payload.is_active else 'deactivated'}",
            "id":         row["id"],
            "is_active":  row["is_active"],
            "updated_at": str(row["updated_at"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  5. Delete a single encoding  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.delete("/api/{subdomain}/employees/{employee_id}/face-encodings/{enc_id}")
def delete_face_encoding(
    subdomain:   str,
    employee_id: str,
    enc_id:      int,
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    DELETE /api/acme/employees/EMP001/face-encodings/3
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        cur.execute("""
            DELETE FROM face_encodings
            WHERE id = %s AND emp_id = %s
            RETURNING id
        """, (enc_id, emp_internal_id))

        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Face encoding #{enc_id} not found for employee '{employee_id}'"
            )

        conn.commit()
        cur.close()

        logging.info(f"[-] Encoding #{enc_id} deleted for '{employee_id}' @ '{subdomain}'")
        return {
            "success": True,
            "message": f"Face encoding #{enc_id} permanently deleted",
            "id":      enc_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  6. Delete ALL encodings for an employee  (Admin only)
# ─────────────────────────────────────────────────────────────────

@app.delete("/api/{subdomain}/employees/{employee_id}/face-encodings")
def delete_all_face_encodings(
    subdomain:   str,
    employee_id: str,
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    Wipe all encodings before re-registering an employee's face.

    DELETE /api/acme/employees/EMP001/face-encodings
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        emp_internal_id = _get_emp_internal_id(cur, employee_id)

        cur.execute(
            "DELETE FROM face_encodings WHERE emp_id = %s RETURNING id",
            (emp_internal_id,)
        )
        deleted_ids = [r["id"] for r in cur.fetchall()]
        conn.commit()
        cur.close()

        logging.info(
            f"[-] All encodings deleted for '{employee_id}' @ '{subdomain}' "
            f"— count: {len(deleted_ids)}"
        )
        return {
            "success":       True,
            "message":       f"All {len(deleted_ids)} encoding(s) deleted for '{employee_id}'",
            "deleted_count": len(deleted_ids),
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  7. Identify face from uploaded image  (Admin / Device)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/{subdomain}/face-encodings/identify")
async def identify_face(
    subdomain: str,
    photo:     UploadFile = File(..., description="Single face image to identify"),
    threshold: float      = Form(
        0.5,
        description="Cosine similarity threshold — same default as THRESHOLD=0.5 in fastapi_server.py"
    ),
    user: CurrentUser = Depends(require_same_subdomain),
):
    """
    Upload one image → identify the employee using cosine similarity.

    Same vectorised numpy matching as fastapi_server.py
    (load_known_face_embeddings_cached + np.dot similarity).

    POST /api/acme/face-encodings/identify
    Header: Authorization: Bearer <token>
    Form:   photo = <file>
            threshold = 0.5   (optional)

    Response (identified):
    {
      "success": true,
      "identified": true,
      "similarity": 0.87,
      "threshold": 0.5,
      "employee": {
        "employee_id": "EMP001",
        "full_name": "John Doe",
        "department": "Engineering",
        "emp_internal_id": 42
      },
      "encoding_id": 7
    }

    Response (not identified):
    {
      "success": true,
      "identified": false,
      "similarity": 0.31,
      "threshold": 0.5,
      "message": "No matching employee found within the given threshold"
    }
    """
    # ── Decode image ──────────────────────────────────────────────
    frame = await _decode_upload(photo)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image (invalid or corrupt file)")

    # ── Detect face ───────────────────────────────────────────────
    faces = face_app.get(frame)

    if not faces:
        return {"success": False, "identified": False, "message": "No face detected in image"}

    if len(faces) > 1:
        return {
            "success":    False,
            "identified": False,
            "message":    f"{len(faces)} faces detected — send a single-face image",
        }

    face = faces[0]

    # ── Try alignment ─────────────────────────────────────────────
    aligned = align_face_improved(frame, face)
    if aligned is not None:
        aligned_faces = face_app.get(aligned)
        if aligned_faces:
            face = aligned_faces[0]

    incoming_emb = face.normed_embedding   # 512-d np.ndarray

    # ── Load all active encodings from DB ─────────────────────────
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT fe.id, fe.emp_id, fe.encoding_vector, fe.confidence_score,
                   e.employee_id, e.full_name, e.department, e.expire_at
            FROM face_encodings fe
            JOIN employees e ON e.id = fe.emp_id
            WHERE fe.is_active = TRUE
              AND (e.expire_at IS NULL OR e.expire_at >= CURRENT_DATE)
        """)
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return {
                "success":    True,
                "identified": False,
                "message":    "No active face encodings registered in this tenant",
            }

        # ── Vectorised cosine similarity (mirrors fastapi_server.py) ─
        stored_embs = []
        for row in rows:
            vec = row["encoding_vector"]
            if isinstance(vec, str):
                vec = json.loads(vec)
            stored_embs.append(vec)

        known_array  = np.array(stored_embs)              # shape (N, 512)
        similarities = np.dot(known_array, incoming_emb) / (
            np.linalg.norm(known_array, axis=1) * np.linalg.norm(incoming_emb)
        )

        best_idx        = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_row        = rows[best_idx]

        if best_similarity >= threshold:
            logging.info(
                f"[✓] Identified → '{best_row['employee_id']}' @ '{subdomain}' "
                f"(similarity={best_similarity:.4f})"
            )
            return {
                "success":    True,
                "identified": True,
                "similarity": round(best_similarity, 4),
                "threshold":  threshold,
                "employee": {
                    "employee_id":     best_row["employee_id"],
                    "full_name":       best_row["full_name"],
                    "department":      best_row["department"],
                    "emp_internal_id": best_row["emp_id"],
                },
                "encoding_id": best_row["id"],
            }

        logging.info(
            f"[?] Not identified @ '{subdomain}' "
            f"best_similarity={best_similarity:.4f} < threshold={threshold}"
        )
        return {
            "success":    True,
            "identified": False,
            "similarity": round(best_similarity, 4),
            "threshold":  threshold,
            "message":    "No matching employee found within the given threshold",
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ identify_face error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  8. Bulk load all active encodings  (Admin — edge device sync)
# ─────────────────────────────────────────────────────────────────

@app.get("/api/{subdomain}/face-encodings/bulk")
def bulk_load_face_encodings(
    subdomain: str,
    user: CurrentUser = Depends(require_admin_same_subdomain),
):
    """
    Returns ALL active encodings with 512-d vectors.
    Use on edge device startup to warm the local cache —
    same concept as load_known_face_embeddings_cached() in fastapi_server.py.

    GET /api/acme/face-encodings/bulk
    Header: Authorization: Bearer <admin_token>
    """
    conn = get_tenant_conn(subdomain)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT fe.id, fe.emp_id, fe.encoding_vector,
                   fe.sample_score, fe.confidence_score,
                   e.employee_id, e.full_name, e.department
            FROM face_encodings fe
            JOIN employees e ON e.id = fe.emp_id
            WHERE fe.is_active = TRUE
              AND (e.expire_at IS NULL OR e.expire_at >= CURRENT_DATE)
            ORDER BY e.employee_id
        """)
        rows = cur.fetchall()
        cur.close()

        return {
            "success":         True,
            "total_encodings": len(rows),
            "face_encodings":  [
                {
                    "encoding_id":     r["id"],
                    "emp_id":          r["emp_id"],
                    "employee_id":     r["employee_id"],
                    "full_name":       r["full_name"],
                    "department":      r["department"],
                    "encoding_vector": r["encoding_vector"],
                    "sample_score":    r["sample_score"],
                    "confidence_score":r["confidence_score"],
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
#  Dependency injection — call this once from main.py / fastapi_server.py
#  so this router shares the SAME face_app instance already loaded,
#  avoiding a second model load and the NameError on import.
#
#  In your fastapi_server.py, after face_app is created, add:
#
#      import face_encodings_routes
#      face_encodings_routes.set_face_app(face_app)
#      app.include_router(face_encodings_routes.router)
# ─────────────────────────────────────────────────────────────────

def set_face_app(app_instance) -> None:
    """Inject the already-initialised FaceAnalysis instance from fastapi_server.py."""
    global face_app
    face_app = app_instance
    logging.info("[face_encodings_routes] face_app injected ✓")


