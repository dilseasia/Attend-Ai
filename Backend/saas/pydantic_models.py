import re
from pydantic import BaseModel, Field, validator
from typing import Optional
from config import PLAN_LIMITS
from datetime import datetime, date
class CompanyRegisterRequest(BaseModel):
    company_name:    str           = Field(..., min_length=2, max_length=255)
    subdomain:       str           = Field(..., min_length=2, max_length=100)
    industry:        Optional[str] = None
    size:            Optional[str] = None
    address:         Optional[str] = None
    admin_full_name: str           = Field(..., min_length=2, max_length=255)
    admin_email:     str           = Field(...)
    admin_phone:     Optional[str] = None
    admin_password:  str           = Field(..., min_length=8)
    plan:            str           = Field("starter")

    @validator("subdomain")
    def validate_subdomain(cls, v):
        if len(v) > 1 and not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', v):
            raise ValueError("Subdomain must be lowercase letters, numbers, and hyphens only")
        if re.search(r'--', v):
            raise ValueError("Subdomain cannot contain consecutive hyphens")
        return v.lower()

    @validator("plan")
    def validate_plan(cls, v):
        if v not in PLAN_LIMITS:
            raise ValueError(f"Plan must be one of: {', '.join(PLAN_LIMITS.keys())}")
        return v

    @validator("admin_email")
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email address")
        return v.lower()

class SubdomainCheckRequest(BaseModel):
    subdomain: str = Field(..., min_length=2, max_length=100)

class PlanUpgradeRequest(BaseModel):
    company_id: int
    new_plan:   str

    # FIXED
    @validator("new_plan")
    def validate_plan(cls, v):
        if v not in PLAN_LIMITS:
            raise ValueError(f"Plan must be one of: {', '.join(PLAN_LIMITS.keys())}")
        return v  # ← must return v
        
class EmployeeRegisterRequest(BaseModel):
    full_name:   str           = Field(..., min_length=2, max_length=255)
    email:       str           = Field(...)
    password:    str           = Field(..., min_length=8)
    department:  Optional[str] = Field(None, max_length=255)
    employee_id: Optional[str] = Field(None, description="Leave blank to auto-generate")
    expire_at:   Optional[str] = Field(None, description="ISO date YYYY-MM-DD, optional account expiry")

    @validator("email")
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email address")
        return v.lower()

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @validator("expire_at")
    def validate_expire_at(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("expire_at must be in YYYY-MM-DD format")


class EmployeeLoginRequest(BaseModel):
    email:    str = Field(...)
    password: str = Field(...)

    @validator("email")
    def lower_email(cls, v):
        return v.lower().strip()


class EmployeeUpdateRequest(BaseModel):
    full_name:  Optional[str] = Field(None, min_length=2, max_length=255)
    department: Optional[str] = Field(None, max_length=255)
    expire_at:  Optional[str] = Field(None, description="YYYY-MM-DD or null to clear")

    @validator("expire_at")
    def validate_expire_at(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("expire_at must be in YYYY-MM-DD format")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(...)
    new_password:     str = Field(..., min_length=8)


class EncodingStatusPatchRequest(BaseModel):
    is_active: bool = Field(..., description="True to activate, False to deactivate")