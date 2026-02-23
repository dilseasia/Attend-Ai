"""
AttendAI – PostgreSQL Database Setup Script
============================================
Creates the master database (attendai_master) with:
  - companies
  - company_databases

Then provisions a tenant database (company_{subdomain}_db) with:
  - admins
  - employees
  - face_encodings
  - recognize_logs
  - attendance_logs
  - anonymous_logs
  - attendance_requests

Usage:
    pip install psycopg2-binary
    python attendai_setup_db.py
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ─────────────────────────────────────────────
#  CONFIG — update these values
# ─────────────────────────────────────────────
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_USER     = "postgres"
DB_PASSWORD = "DALJEET123"

MASTER_DB_NAME = "attendai_master"
# ─────────────────────────────────────────────


def get_connection(dbname="postgres"):
    """Connect to a given database (defaults to postgres for admin tasks)."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=dbname,
    )


def create_database(db_name: str):
    """Create a database if it doesn't already exist."""
    conn = get_connection()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()

    if exists:
        print(f"  [~] Database '{db_name}' already exists — skipping creation.")
    else:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"  [+] Database '{db_name}' created.")

    cur.close()
    conn.close()


# ─────────────────────────────────────────────
#  MASTER DB TABLES
# ─────────────────────────────────────────────

MASTER_TABLES_SQL = """

-- ── companies ────────────────────────────────
CREATE TABLE IF NOT EXISTS companies (
    id                      BIGSERIAL PRIMARY KEY,
    name                    VARCHAR(255)    NOT NULL,
    subdomain               VARCHAR(100)    NOT NULL UNIQUE,
    plan                    VARCHAR(50)     NOT NULL DEFAULT 'free'
                                CHECK (plan IN ('free', 'starter', 'pro', 'enterprise')),
    is_active_subscription  BOOLEAN         NOT NULL DEFAULT TRUE,
    max_employees           INT             NOT NULL DEFAULT 50,
    created_at              TIMESTAMP       NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMP
);

-- ── company_databases (routing table) ────────
CREATE TABLE IF NOT EXISTS company_databases (
    id              BIGSERIAL PRIMARY KEY,
    company_id      BIGINT      NOT NULL UNIQUE REFERENCES companies(id) ON DELETE CASCADE,
    db_name         VARCHAR(255) NOT NULL UNIQUE,
    db_host         VARCHAR(255) NOT NULL DEFAULT 'localhost',
    db_port         INT          NOT NULL DEFAULT 5432,
    db_user         VARCHAR(255) NOT NULL,
    db_password_enc VARCHAR(500) NOT NULL,
    is_provisioned  BOOLEAN      NOT NULL DEFAULT FALSE,
    provisioned_at  TIMESTAMP
);

"""


def setup_master_db():
    print("\n[1] Setting up Master Database...")
    create_database(MASTER_DB_NAME)

    conn = get_connection(MASTER_DB_NAME)
    cur = conn.cursor()

    cur.execute(MASTER_TABLES_SQL)
    conn.commit()

    print("  [+] Table 'companies' — OK")
    print("  [+] Table 'company_databases' — OK")

    cur.close()
    conn.close()
    print(f"  ✅ Master DB '{MASTER_DB_NAME}' ready.\n")


# ─────────────────────────────────────────────
#  TENANT DB TABLES
# ─────────────────────────────────────────────

TENANT_TABLES_SQL = """

-- ── admins ───────────────────────────────────
CREATE TABLE IF NOT EXISTS admins (
    id          BIGSERIAL PRIMARY KEY,
    email       VARCHAR(255)    NOT NULL UNIQUE,
    password    VARCHAR(500)    NOT NULL,
    full_name   VARCHAR(255)    NOT NULL,
    is_active   BOOLEAN         NOT NULL DEFAULT TRUE,
    last_login  TIMESTAMP,
    created_at  TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ── employees ────────────────────────────────
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

-- ── face_encodings ───────────────────────────
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

-- ── recognize_logs ───────────────────────────
CREATE TABLE IF NOT EXISTS recognize_logs (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    date             DATE        NOT NULL DEFAULT CURRENT_DATE,
    entry_photos     VARCHAR(500),
    exit_photos      VARCHAR(500),
    confidence_score FLOAT
);

-- ── attendance_logs ──────────────────────────
CREATE TABLE IF NOT EXISTS attendance_logs (
    id             BIGSERIAL PRIMARY KEY,
    user_id        BIGINT      NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    date           DATE        NOT NULL DEFAULT CURRENT_DATE,
    time           TIME        NOT NULL DEFAULT CURRENT_TIME,
    entry_or_exit  VARCHAR(10) NOT NULL CHECK (entry_or_exit IN ('entry', 'exit')),
    device_id      VARCHAR(255)
);

-- ── anonymous_logs ───────────────────────────
CREATE TABLE IF NOT EXISTS anonymous_logs (
    id            BIGSERIAL PRIMARY KEY,
    date          DATE        NOT NULL DEFAULT CURRENT_DATE,
    entry         TIME,
    exit          TIME,
    photo_capture VARCHAR(500)
);

-- ── attendance_requests ──────────────────────
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


def provision_tenant_db(subdomain: str):
    """
    Creates a tenant database for the given company subdomain
    and sets up all 7 tenant tables inside it.

    DB name format: company_{subdomain}_db
    """
    tenant_db_name = f"company_{subdomain}_db"
    print(f"[2] Provisioning Tenant Database for '{subdomain}'...")

    create_database(tenant_db_name)

    conn = get_connection(tenant_db_name)
    cur = conn.cursor()

    cur.execute(TENANT_TABLES_SQL)
    conn.commit()

    tables = [
        "admins", "employees", "face_encodings",
        "recognize_logs", "attendance_logs",
        "anonymous_logs", "attendance_requests"
    ]
    for t in tables:
        print(f"  [+] Table '{t}' — OK")

    cur.close()
    conn.close()
    print(f"  ✅ Tenant DB '{tenant_db_name}' ready.\n")

    # Register in master DB routing table
    register_tenant_in_master(subdomain, tenant_db_name)


def register_tenant_in_master(subdomain: str, tenant_db_name: str):
    """
    Registers the tenant DB connection info into company_databases
    in the master DB (looks up company by subdomain).
    """
    conn = get_connection(MASTER_DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT id FROM companies WHERE subdomain = %s", (subdomain,))
    row = cur.fetchone()

    if not row:
        print(f"  [!] No company found with subdomain '{subdomain}' in master DB.")
        print(f"      Skipping routing table registration.\n")
        cur.close()
        conn.close()
        return

    company_id = row[0]

    cur.execute("""
        INSERT INTO company_databases
            (company_id, db_name, db_host, db_port, db_user, db_password_enc, is_provisioned, provisioned_at)
        VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW())
        ON CONFLICT (company_id) DO UPDATE
            SET is_provisioned = TRUE,
                provisioned_at = NOW();
    """, (company_id, tenant_db_name, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD))

    conn.commit()
    print(f"  [+] Registered '{tenant_db_name}' in master routing table.")
    cur.close()
    conn.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AttendAI — PostgreSQL Setup Script")
    print("=" * 50)

    # Step 1: Always create the master DB
    setup_master_db()

    # Step 2 (Optional): Provision a tenant DB for a specific company.
    # Uncomment the lines below and set the subdomain after adding the
    # company row to the 'companies' table in the master DB.
    #
    # Example:
    #   subdomain = "acme"
    #   provision_tenant_db(subdomain)

    print("Done! Master DB is ready.")
    print("To provision a tenant DB, call: provision_tenant_db('your_subdomain')")
    print("=" * 50)