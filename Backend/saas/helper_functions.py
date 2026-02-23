import psycopg2
import hashlib
import logging
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Optional
from fastapi import HTTPException
from config import MASTER_DB_CONFIG, POSTGRES_ADMIN_CONFIG
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_master_connection():
    return psycopg2.connect(**MASTER_DB_CONFIG)


def create_database_if_not_exists(db_name: str) -> bool:
    conn = psycopg2.connect(**POSTGRES_ADMIN_CONFIG)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if cur.fetchone():
            logging.info(f"[~] DB '{db_name}' already exists")
            return False
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        logging.info(f"[+] DB '{db_name}' created")
        return True
    finally:
        cur.close()
        conn.close()

TENANT_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS admins (
    id          BIGSERIAL PRIMARY KEY,
    email       VARCHAR(255)    NOT NULL UNIQUE,
    password    VARCHAR(500)    NOT NULL,
    full_name   VARCHAR(255)    NOT NULL,
    phone       VARCHAR(50),
    is_active   BOOLEAN         NOT NULL DEFAULT TRUE,
    last_login  TIMESTAMP,
    created_at  TIMESTAMP       NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS employees (
    id          BIGSERIAL PRIMARY KEY,
    employee_id VARCHAR(100)    NOT NULL UNIQUE,
    full_name   VARCHAR(255)    NOT NULL,
    email       VARCHAR(255)    NOT NULL UNIQUE,
    password    VARCHAR(500)    NOT NULL,
    department  VARCHAR(255),
    created_at  TIMESTAMP       NOT NULL DEFAULT NOW(),
    expire_at   TIMESTAMP
);
CREATE TABLE IF NOT EXISTS face_encodings (
    id               BIGSERIAL PRIMARY KEY,
    emp_id           BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    encoding_vector  JSONB       NOT NULL,
    sample_score     FLOAT,
    confidence_score FLOAT,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMP   NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP   NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS recognize_logs (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    date             DATE        NOT NULL DEFAULT CURRENT_DATE,
    entry_photos     VARCHAR(500),
    exit_photos      VARCHAR(500),
    confidence_score FLOAT
);
CREATE TABLE IF NOT EXISTS attendance_logs (
    id             BIGSERIAL PRIMARY KEY,
    user_id        BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    date           DATE        NOT NULL DEFAULT CURRENT_DATE,
    time           TIME        NOT NULL DEFAULT CURRENT_TIME,
    entry_or_exit  VARCHAR(10) NOT NULL CHECK (entry_or_exit IN ('entry', 'exit')),
    device_id      VARCHAR(255)
);
CREATE TABLE IF NOT EXISTS anonymous_logs (
    id            BIGSERIAL PRIMARY KEY,
    date          DATE        NOT NULL DEFAULT CURRENT_DATE,
    entry         TIME,
    exit          TIME,
    photo_capture VARCHAR(500)
);
CREATE TABLE IF NOT EXISTS attendance_requests (
    id            BIGSERIAL PRIMARY KEY,
    emp_id        BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    request_type  VARCHAR(50) NOT NULL CHECK (request_type IN ('regularize', 'leave', 'wfh', 'other')),
    date          DATE        NOT NULL,
    in_time       TIME,
    out_time      TIME,
    reason        TEXT,
    status        VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
    requested_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    approved_at   TIMESTAMP,
    approved_by   BIGINT      REFERENCES admins(id) ON DELETE SET NULL,
    remarks       TEXT
);
"""

def provision_tenant_database(subdomain: str) -> str:
    db_name = f"company_{subdomain}_db"
    create_database_if_not_exists(db_name)
    tenant_config = {**POSTGRES_ADMIN_CONFIG, "dbname": db_name}
    conn = psycopg2.connect(**tenant_config)
    cur = conn.cursor()
    try:
        cur.execute(TENANT_TABLES_SQL)
        conn.commit()
        logging.info(f"[+] Tenant tables created in '{db_name}'")
    finally:
        cur.close()
        conn.close()
    return db_name

def register_company_in_master(company_name, subdomain, plan, max_employees, db_name) -> int:
    conn = get_master_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO companies (name, subdomain, plan, max_employees, is_active_subscription, created_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW()) RETURNING id
        """, (company_name, subdomain, plan, max_employees))
        company_id = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO company_databases
                (company_id, db_name, db_host, db_port, db_user, db_password_enc, is_provisioned, provisioned_at)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW())
            ON CONFLICT (company_id) DO UPDATE SET is_provisioned = TRUE, provisioned_at = NOW()
        """, (company_id, db_name, POSTGRES_ADMIN_CONFIG["host"], POSTGRES_ADMIN_CONFIG["port"],
              POSTGRES_ADMIN_CONFIG["user"], POSTGRES_ADMIN_CONFIG["password"]))
        conn.commit()
        return company_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def create_admin_in_tenant(subdomain, full_name, email, password, phone=None) -> int:
    db_name = f"company_{subdomain}_db"
    tenant_config = {**POSTGRES_ADMIN_CONFIG, "dbname": db_name}
    conn = psycopg2.connect(**tenant_config)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO admins (email, password, full_name, phone, is_active, created_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW()) RETURNING id
        """, (email, hash_password(password), full_name, phone))
        admin_id = cur.fetchone()[0]
        conn.commit()
        return admin_id
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"Admin email '{email}' already exists")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def hash_password(password: str) -> str:
    """SHA-256 hash. Replace with bcrypt in production."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed


def get_tenant_conn(subdomain: str):
    """Connect to a company's tenant DB."""
    db_name = f"company_{subdomain.lower()}_db"
    try:
        conn = psycopg2.connect(**{**POSTGRES_ADMIN_CONFIG, "dbname": db_name})
        return conn
    except psycopg2.OperationalError:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{subdomain}' not found or database not provisioned"
        )
        pritnt("Error connecting to tenant DB:", e)


def generate_employee_id(subdomain: str, conn) -> str:
    """Auto-generate employee ID like EMP-0001, EMP-0002..."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM employees")
    count = cur.fetchone()[0]
    cur.close()
    prefix = subdomain[:3].upper()
    return f"{prefix}-{str(count + 1).zfill(4)}"


def check_employee_limit(subdomain: str, conn):
    """Check if company has hit its plan's employee limit."""
    from config import MASTER_DB_CONFIG
    # Get plan limit from master DB
    try:
        master_conn = psycopg2.connect(**MASTER_DB_CONFIG)
        cur = master_conn.cursor()
        cur.execute("SELECT max_employees, plan FROM companies WHERE subdomain = %s", (subdomain.lower(),))
        row = cur.fetchone()
        cur.close()
        master_conn.close()
        if not row:
            return  # company not found, let it proceed
        max_employees, plan = row
    except Exception:
        return  # fail open if master DB unreachable

    # Count current employees in tenant DB
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM employees")
    current = cur.fetchone()[0]
    cur.close()

    if current >= max_employees:
        raise HTTPException(
            status_code=403,
            detail=f"Employee limit reached for '{plan}' plan ({max_employees} max). Please upgrade."
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Same function as fastapi_server.py"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def align_face_improved(frame: np.ndarray, face) -> Optional[np.ndarray]:
    """Same alignment logic as fastapi_server.py"""
    try:
        if not hasattr(face, "kps") or face.kps is None:
            return None

        kps = face.kps
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(kps, ref_pts)[0]
        if tform is None:
            return None

        return cv2.warpAffine(frame, tform, (112, 112))

    except Exception as e:
        logging.debug(f"Face alignment failed: {e}")
        return None


def _compute_sample_score(face, frame_shape: tuple) -> float:
    """
    Derive image quality score (0-1):
      70% weight → det_score  (InsightFace detection confidence)
      30% weight → bbox area ratio vs frame (bigger face = better sample)
    """
    det_score = float(face.det_score) if hasattr(face, "det_score") else 0.5

    try:
        bbox       = face.bbox   # [x1, y1, x2, y2]
        face_area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = min(float(face_area) / float(frame_area), 1.0)
        score      = 0.7 * det_score + 0.3 * area_ratio
    except Exception:
        score = det_score

    return round(float(score), 4)


def _get_emp_internal_id(cur, employee_id: str) -> int:
    """Resolve employee_id string → internal DB id"""
    cur.execute("SELECT id FROM employees WHERE employee_id = %s", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
    return row["id"] if isinstance(row, dict) else row[0]


async def _decode_upload(upload):
    import io
    import numpy as np
    import cv2
    from PIL import Image

    await upload.seek(0)
    data = await upload.read()

    if not data:
        return None

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        import logging
        logging.error(f"[decode] Failed '{upload.filename}': {e}")
        return None