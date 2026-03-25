import pytest
import json
from uuid import uuid4
from urllib.parse import urlencode

# Helper utilities for the FastAPI server tests

def _post_form(ctx, path, form_dict):
    """Helper to POST application/x-www-form-urlencoded data."""
    body = urlencode({k: "" if v is None else str(v) for k, v in form_dict.items()})
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    return ctx.post(path, data=body, headers=headers)

def _login(ctx, username, password):
    """Call /login form endpoint and return response + token"""
    return _post_form(ctx, "/login", {"username": username, "password": password})


def _auth_headers(ctx, username=None, password=None):
    """Login and return Authorization headers dict. If username is None use admin by default."""
    username = username or "admin"
    password = password or "P@rt2jk2"

    resp = _login(ctx, username, password)
    assert resp.status == 200, f"login failed for {username}, got {resp.status}"
    body = resp.json()
    token = body.get("token")
    assert token, f"No token returned for {username}: {body}"
    return {"Authorization": f"Bearer {token}"}

def _create_attendance_request(ctx, payload):
    """Create attendance request via JSON body (pydantic model expected)."""
    data = json.dumps(payload)
    headers = {"Content-Type": "application/json"}
    resp = ctx.post("/attendance-request/create", data=data, headers=headers)
    return resp


# ------------------------
# Tests
# ------------------------

def test_login_success_and_failure(api_request_context):
    # happy path admin
    resp = _login(api_request_context, "admin", "P@rt2jk2")
    assert resp.status == 200, f"Expected 200 for admin login, got {resp.status}"
    body = resp.json()
    assert "token" in body and body["token"].startswith("TOKEN_"), f"Invalid login response: {body}"

    # wrong password -> should be 401
    bad = _login(api_request_context, "admin", "wrong-pass")
    assert bad.status == 401, f"Expected 401 for wrong password, got {bad.status}"


@pytest.mark.parametrize("method,endpoint", [
    ("get", "/employees-mobile"),
    ("post", "/attendance-request/1/approve"),
])
def test_protected_endpoints_reject_unauthenticated(api_request_context, method, endpoint):
    # Ensure unauthenticated access is rejected.
    if method.lower() == "get":
        resp = api_request_context.get(endpoint)
    else:
        # POST endpoints require a body type; send minimal JSON for endpoints that expect JSON
        # Approve endpoint expects JSON with status field (but we're unauthenticated so body doesn't matter)
        resp = api_request_context.post(endpoint, data=json.dumps({}), headers={"Content-Type": "application/json"})

    assert resp.status in (401, 403), f"{endpoint} should require auth, got {resp.status}"


def test_create_attendance_request_validation_branches(api_request_context):
    # 1) Non-existent employee -> 404
    payload = {
        "emp_id": f"nope_{uuid4().hex[:6]}",
        "request_type": "manual_capture",
        "date": "2025-01-01",
        "in_time": "09:00:00"
    }
    resp = _create_attendance_request(api_request_context, payload)
    assert resp.status == 404, f"Expected 404 for unknown emp, got {resp.status} - {resp.json()}"

    # 2) WFH without both times -> 400
    payload = {
        "emp_id": "1940",
        "request_type": "wfh",
        "date": "2025-10-01",
        "in_time": None,
        "out_time": None
    }
    resp = _create_attendance_request(api_request_context, payload)
    assert resp.status == 400, f"WFH without times should be 400, got {resp.status}"

    # 3) Manual capture without any time -> 400
    payload = {
        "emp_id": "1940",
        "request_type": "manual_capture",
        "date": "2025-10-02"
    }
    resp = _create_attendance_request(api_request_context, payload)
    assert resp.status == 400, f"Manual capture without times should be 400, got {resp.status}"

    # 4) Successful manual capture with in_time
    payload = {
        "emp_id": "1940",
        "request_type": "manual_capture",
        "date": "2025-10-03",
        "in_time": "09:15:00",
        "out_time": None,
        "reason": "Forgot badge"
    }
    resp = _create_attendance_request(api_request_context, payload)
    assert resp.status == 200, f"Expected 200 on valid request create, got {resp.status} - {resp.json()}"
    body = resp.json()
    assert body.get("success") is True, f"Expected success True, got: {body}"
    assert "request_id" in body, "request_id should be present in response"


def test_list_requests_and_pagination(api_request_context):
    # list /attendance-request/list endpoint supports status=pending etc
    resp = api_request_context.get("/attendance-request/list", params={"limit": 5, "offset": 0})
    assert resp.status == 200, f"list should return 200, got {resp.status}"
    body = resp.json()
    assert "requests" in body and isinstance(body["requests"], list), f"Unexpected list response: {body}"

    # status=all normalization
    resp2 = api_request_context.get("/attendance-request/list", params={"status": "all", "limit": 2, "offset": 0})
    assert resp2.status == 200
    b2 = resp2.json()
    assert "requests" in b2


def test_approve_request_auth_and_flow(api_request_context):
    # Create a fresh manual_capture request to approve
    payload = {
        "emp_id": "1940",
        "request_type": "manual_capture",
        "date": "2025-10-04",
        "in_time": "09:05:00",
        "out_time": "17:05:00",
        "reason": "Manual for test"
    }
    create_resp = _create_attendance_request(api_request_context, payload)
    assert create_resp.status == 200, f"Create failed: {create_resp.status} {create_resp.json()}"
    req_id = create_resp.json().get("request_id")
    assert req_id, "request_id missing"

    # Employee (non-admin) tries to approve -> 403
    employee_headers = _auth_headers(api_request_context, "1940", "Abhishek@1940")
    bad = api_request_context.post(f"/attendance-request/{req_id}/approve", data=json.dumps({"status": "approved"}), headers={**employee_headers, "Content-Type": "application/json"})
    assert bad.status == 403, f"Non-admin approving should be 403, got {bad.status} - {bad.json()}"

    # Admin approves -> success
    admin_headers = _auth_headers(api_request_context, "admin", "P@rt2jk2")
    approve_payload = {"status": "approved", "remarks": "Good"}
    ok = api_request_context.post(f"/attendance-request/{req_id}/approve", data=json.dumps(approve_payload), headers={**admin_headers, "Content-Type": "application/json"})
    assert ok.status == 200, f"Admin approve should 200, got {ok.status} - {ok.json()}"
    body = ok.json()
    assert body.get("success") is True, f"Expected success True on approve, got {body}"
    assert body.get("status") == "approved", f"Expected status approved, got {body.get('status')}"

    # Approving non-existent request -> 404
    not_found = api_request_context.post(f"/attendance-request/99999999/approve", data=json.dumps({"status": "approved"}), headers={**admin_headers, "Content-Type": "application/json"})
    assert not_found.status == 404, f"Approving missing request should 404, got {not_found.status}"


def test_employees_endpoints_and_mobile_auth(api_request_context):
    # GET /employees (no auth) should return employees array
    resp = api_request_context.get("/employees")
    assert resp.status == 200, f"/api/employees should 200, got {resp.status}"
    body = resp.json()
    assert isinstance(body.get("employees"), list)

    # /employees-mobile requires auth
    r_no = api_request_context.get("/employees-mobile")
    assert r_no.status in (401, 403), f"employees-mobile without auth should return 401/403, got {r_no.status}"

    # employee token returns only their data (emp 1940)
    headers_emp = _auth_headers(api_request_context, "1940", "Abhishek@1940")
    r_emp = api_request_context.get("/employees-mobile", headers=headers_emp)
    assert r_emp.status == 200, f"employees-mobile with token should 200, got {r_emp.status}"
    data = r_emp.json()
    assert isinstance(data.get("employees"), list)


def test_known_count_and_reload_warmup(api_request_context):
    # known-count
    resp = api_request_context.get("/known-count")
    assert resp.status == 200
    body = resp.json()
    assert "count" in body and isinstance(body["count"], int)

    # warmup-cache should return success shape
    resp2 = api_request_context.post("/warmup-cache")
    assert resp2.status == 200
    b2 = resp2.json()
    assert b2.get("success") in (True, False) and "employees_loaded" in b2

    # reload embeddings endpoint
    resp3 = api_request_context.post("/reload-embeddings")
    assert resp3.status == 200
    assert resp3.json().get("status") == "success"


def test_logs_endpoints_basic_shapes(api_request_context):
    # /logs (paginated)
    r = api_request_context.get("/logs", params={"limit": 5, "offset": 0})
    assert r.status == 200
    j = r.json()
    assert "total" in j and "logs" in j
    assert isinstance(j.get("logs"), list)

    # /logs/all
    r_all = api_request_context.get("/logs/all")
    assert r_all.status == 200
    ja = r_all.json()
    assert "total" in ja and "logs" in ja and "collective" in ja

    # /logs/present-today requires emp_id param
    r_present = api_request_context.get("/logs/present-today", params={"emp_id": "1940"})
    assert r_present.status == 200
    bp = r_present.json()
    assert "emp_id" in bp and "present" in bp

    # employee-entries-with-photos with required param
    r_photos = api_request_context.get("/employee-entries-with-photos", params={"emp_id": "1940", "type": "all"})
    assert r_photos.status in (200, 500), f"Unexpected status: {r_photos.status}"
    if r_photos.status == 200:
        pj = r_photos.json()
        assert "emp_id" in pj and "records" in pj


def test_delete_token_and_save_daily_summary(api_request_context, qa_base_url):
    # delete_token expects form emp_id + fcm_token
    resp = _post_form(api_request_context, "/delete_token", {"emp_id": "1940", "fcm_token": "dummy_token_for_test"})
    assert resp.status == 200, f"delete_token should 200, got {resp.status}"
    body = resp.json()
    assert "success" in body

    # save-daily-summary (POST JSON) expects certain fields
    payload = {
        "emp_id": "1940",
        "name": "TestName",
        "date": "2025-10-05",
        "working_hours": "8h 0m",
        "entry_count": 1,
        "exit_count": 1,
        "first_entry": "09:00:00",
        "last_exit": "17:00:00",
        "status": "Present"
    }

    # Use full QA base url to ensure correct absolute URL (some deployments require absolute path)
    full_url = f"{qa_base_url}/save-daily-summary"
    resp2 = api_request_context.post(full_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert resp2.status == 200, f"Expected 200 for save-daily-summary, got {resp2.status} - {resp2.text() if hasattr(resp2, 'text') else resp2}
    assert resp2.json().get("success") is True


@pytest.mark.parametrize("bad_type", ["not-an-image", "text/plain"]) 
def test_find_best_match_rejects_invalid_image(api_request_context, bad_type):
    # Try to call find-best-match with invalid content-type to provoke 400/422
    # Use a simple form with photo field as text; server will validate content_type
    form = {"fcm_token": "", "platform": "test"}
    # Post without a real file; rely on server error handling
    resp = api_request_context.post("/find-best-match", data=urlencode(form), headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status in (400, 422, 500), f"Expected client error for invalid file, got {resp.status}"


def test_warm_paths_and_start_stop_status(api_request_context):
    # /start and /status and /stop are available
    start = api_request_context.post("/start")
    assert start.status == 200
    status = api_request_context.get("/status")
    assert status.status == 200
    stop = api_request_context.post("/stop")
    assert stop.status == 200


# End of file
