from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException,
    Request, Query, Body, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, shutil, subprocess, cv2, re
from datetime import datetime
import psycopg2
import time
from psycopg2.extras import RealDictCursor
from attendance_db_postgres import DB_CONFIG, get_connection
from datetime import datetime, timedelta
from fastapi import APIRouter, Body, HTTPException
import os
import time 
import shutil
from fastapi import FastAPI, Query, HTTPException
import psycopg2
from datetime import datetime
from psycopg2.extras import RealDictCursor
import calendar
from fastapi import Query
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict
from fastapi import Query
from datetime import datetime, date as dt_date, timedelta
import calendar
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from fastapi import Header, Depends
from fastapi import Depends, HTTPException
import os
from fastapi import Query
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from fastapi import Depends
import face_recognition
import numpy as np
from fastapi import UploadFile, File, Form, HTTPException
import os
import io
from PIL import Image
import face_recognition
import numpy as np
from fastapi import UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
import os
import io
from PIL import Image
from fastapi import UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import logging
#  PostgreSQL Database Functions
from attendance_db_postgres import (
    log_attendance,
    init_db,
    init_summary_table,
    save_daily_summary,
    get_daily_summary,
)

from datetime import datetime, time as dt_time
from typing import Optional
from pydantic import BaseModel, Field, validator

# Import the new database functions (add after attendance_db_postgres imports)
from attendance_db_postgres import (
    init_attendance_requests_table,
    create_attendance_request,
    get_attendance_requests,
    update_request_status
)

security = HTTPBearer()



app = FastAPI(title="Face Attendance Server")

#  Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Initialize PostgreSQL tables
init_db()
init_summary_table()


USERS = {
    "admin": {
        "password": "P@rt2jk2",
        "role": "admin"
    },

    "1940": {"password": "Abhishek@1940", "role": "employee"},
    "2426": {"password": "Amandeep@2426", "role": "employee"},
    "2741": {"password": "Amandeep@2741", "role": "employee"},
    "1223": {"password": "Amandeep@1223", "role": "employee"},
    "2675": {"password": "Amarjeet@2675", "role": "employee"},
    "1253": {"password": "Anil@1253", "role": "employee"},
    "4007": {"password": "Ankit@4007", "role": "employee"},
    "2585": {"password": "Anuj@2585", "role": "employee"},
    "1631": {"password": "Arun@1631", "role": "employee"},
    "2022": {"password": "Ashish@2022", "role": "employee"},
    "2572": {"password": "Ganesh@2572", "role": "employee"},
    "1785": {"password": "Gurinderpal@1785", "role": "employee"},
    "2075": {"password": "Gurkiran@2075", "role": "employee"},
    "2080": {"password": "Gurpreet@2080", "role": "employee"},
    "2561": {"password": "Harjodh@2561", "role": "employee"},
    "2687": {"password": "Harpreet@2687", "role": "employee"},
    "1260": {"password": "Jatinder@1260", "role": "employee"},
    "1763": {"password": "Kanika@1763", "role": "employee"},
    "1851": {"password": "Panket@1851", "role": "employee"},
    "3940": {"password": "Pavitar@3940", "role": "employee"},
    "2012": {"password": "Priyank@2012", "role": "employee"},
    "4046": {"password": "Rachita@4046", "role": "employee"},
    "3893": {"password": "Rajeev@3893", "role": "employee"},
    "2549": {"password": "Ramandeep@2549", "role": "employee"},
    "1100": {"password": "Ritu@1100", "role": "employee"},
    "3087": {"password": "Sandeep@3087", "role": "employee"},
    "1122": {"password": "Suresh@1122", "role": "employee"},
    "5022": {"password": "Suresh@5022", "role": "employee"},
    "3105": {"password": "Tushar@3105", "role": "employee"},
    "2984": {"password": "Vivek@2984", "role": "employee"},
    "2567": {"password": "Vivek@2567", "role": "employee"},
    "1131": {"password": "Yashu@1131", "role": "employee"},
}


# Initialize InsightFace (same as entry.py)
# This should be done ONCE at the top of fastapi_server.py
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

THRESHOLD = 0.5  # Same as entry.py

def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings (same as entry.py)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def align_face_improved(frame, face):
    """
    Align face using affine transformation (same as entry.py)
    """
    try:
        if not hasattr(face, 'kps') or face.kps is None:
            return None
            
        kps = face.kps
        
        # Reference points for aligned face (112x112 output)
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        tform = cv2.estimateAffinePartial2D(kps, ref_pts)[0]
        
        if tform is None:
            return None
            
        aligned = cv2.warpAffine(frame, tform, (112, 112))
        return aligned
        
    except Exception as e:
        logging.debug(f"Face alignment failed: {e} - fastapi_server.py:177")
        return None


# ============================================
# GLOBAL CACHE (Load once at startup)
# ============================================
KNOWN_FACES_CACHE = {}
KNOWN_EMBEDDINGS_CACHE = []
CACHE_LOADED = False

def load_known_face_embeddings_cached():
    """
    Load embeddings ONCE and cache them globally.
    Returns: dict of face info, list of embeddings
    """
    global KNOWN_FACES_CACHE, KNOWN_EMBEDDINGS_CACHE, CACHE_LOADED
    
    # If already cached, return immediately
    if CACHE_LOADED:
        return KNOWN_FACES_CACHE, KNOWN_EMBEDDINGS_CACHE
    
    known_faces = {}
    known_embeddings = []
    base_url = "http://10.8.11.183:8000"
    
    if not os.path.exists("known_faces"):
        logging.warning("⚠️ Known faces directory not found")
        return known_faces, known_embeddings
    
    logging.info("🔄 Loading known face embeddings (one-time operation)...")
    
    for folder in os.listdir("known_faces"):
        if "_" not in folder:
            continue
        
        try:
            name, emp_id = folder.rsplit("_", 1)
        except ValueError:
            logging.warning(f"⚠️ Invalid folder format: {folder}")
            continue
        
        folder_path = os.path.join("known_faces", folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Find profile photo
        profile_photo = None
        for file in os.listdir(folder_path):
            if (file.lower().endswith(('.jpg', '.jpeg', '.png')) 
                and not file.startswith("auto_") 
                and not file.endswith(".npy")):
                profile_photo = f"{base_url}/known_faces/{folder}/{file}"
                break
        
        # Load face images and extract embeddings
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if file.startswith("auto_") or file.endswith(".npy"):
                continue
            
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                faces = face_app.get(img)
                if faces:
                    idx = len(known_embeddings)
                    known_faces[idx] = {
                        "name": name,
                        "emp_id": emp_id,
                        "profile_photo": profile_photo
                    }
                    known_embeddings.append(faces[0].embedding)
                    
            except Exception as e:
                logging.error(f"❌ Error loading {img_path}: {e}")
                continue
    
    # Update global cache
    KNOWN_FACES_CACHE = known_faces
    KNOWN_EMBEDDINGS_CACHE = known_embeddings
    CACHE_LOADED = True
    
    logging.info(f"✅ Cached {len(known_embeddings)} face embeddings")
    return known_faces, known_embeddings




def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials  # TOKEN_1940

    username = token.replace("TOKEN_", "")
    user = USERS.get(username)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "username": username,
        "role": user["role"]
    }

@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    user = USERS.get(username)

    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = f"TOKEN_{username}"

    return {
        "token": token,
        "role": user["role"],
        "username": username
    }



@app.on_event("startup")
async def startup_event():
    """Pre-load embeddings when server starts"""
    logging.info("🚀 Server starting - loading face embeddings...")
    load_known_face_embeddings_cached()
    logging.info("✅ Face embeddings loaded and cached")
    
    # Initialize attendance requests table
    init_attendance_requests_table()
    logging.info("✅ Attendance requests table initialized")

# ============================================
# LOAD CACHE AT STARTUP (runs once)
# ============================================
# @app.on_event("startup")
# async def startup_event():
#     """Pre-load embeddings when server starts"""
#     logging.info("🚀 Server starting - loading face embeddings...")
#     load_known_face_embeddings_cached()
#     logging.info("✅ Face embeddings loaded and cached")





# ============================================
# PYDANTIC MODELS FOR REQUEST VALIDATION
# ============================================

class AttendanceRequestCreate(BaseModel):
    emp_id: str = Field(..., description="Employee ID")
    request_type: str = Field(..., description="Type: 'wfh' or 'manual_capture'")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    in_time: Optional[str] = Field(None, description="Check-in time (HH:MM or HH:MM:SS)")
    out_time: Optional[str] = Field(None, description="Check-out time (HH:MM or HH:MM:SS)")
    reason: Optional[str] = Field(None, description="Reason for request")
    
    @validator('request_type')
    def validate_request_type(cls, v):
        if v not in ['wfh', 'manual_capture']:
            raise ValueError("request_type must be 'wfh' or 'manual_capture'")
        return v
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v
    
    @validator('in_time', 'out_time', pre=True)
    def validate_time(cls, v):
        # Allow None, empty string, or placeholder values like "string"
        if v is None or v == "" or v == "string":
            return None
        
        try:
            # Try parsing with seconds
            datetime.strptime(v, '%H:%M:%S')
        except ValueError:
            try:
                # Try parsing without seconds
                datetime.strptime(v, '%H:%M')
            except ValueError:
                raise ValueError("time must be in HH:MM or HH:MM:SS format")
        return v


class RequestApproval(BaseModel):
    status: str = Field(..., description="'approved' or 'rejected'")
    remarks: Optional[str] = Field(None, description="Approval/rejection remarks")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ['approved', 'rejected']:
            raise ValueError("status must be 'approved' or 'rejected'")
        return v




# ============================================
# API ENDPOINTS (NO AUTHORIZATION)
# ============================================
@app.post("/api/attendance-request/create")
async def create_request(request: AttendanceRequestCreate):
    """
    Create a new attendance request (WFH or Manual Capture).

    No authorization checks.
    Reason is optional.
    
    Validation rules:
    - WFH: Both in_time and out_time are required
    - Manual Capture: At least one of in_time or out_time is required
    """
    try:
        # Get employee name from known_faces folder
        emp_name = None
        if os.path.exists("known_faces"):
            for folder in os.listdir("known_faces"):
                if folder.endswith(f"_{request.emp_id}"):
                    emp_name = folder.split("_")[0]
                    break

        if not emp_name:
            raise HTTPException(
                status_code=404,
                detail=f"Employee {request.emp_id} not found in system"
            )

        # Validate based on request type
        if request.request_type == "wfh":
            # WFH requires both in_time and out_time
            if not request.in_time or not request.out_time:
                raise HTTPException(
                    status_code=400,
                    detail="Both in_time and out_time are required for WFH requests"
                )
        
        elif request.request_type == "manual_capture":
            # Manual capture requires at least one timing
            if not request.in_time and not request.out_time:
                raise HTTPException(
                    status_code=400,
                    detail="At least one of in_time or out_time is required for manual capture"
                )

        request_id = create_attendance_request(
            emp_id=request.emp_id,
            name=emp_name,
            request_type=request.request_type,
            date=request.date,
            in_time=request.in_time,
            out_time=request.out_time,
            reason=request.reason  # ✅ can be None
        )

        return {
            "success": True,
            "message": "Attendance request created successfully",
            "request_id": request_id,
            "request_type": request.request_type,
            "status": "pending"
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Error creating attendance request: {e} - fastapi_server.py:479")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create attendance request: {str(e)}"
        )



@app.get("/api/attendance-request/list")
async def list_requests(
    status: Optional[str] = Query(
        None,
        description="pending / approved / rejected / all"
    ),
    emp_id: Optional[str] = Query(None, description="Employee ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get list of attendance requests.

    - status=all returns all
    - reason included only if present, else null
    """
    try:
        # Normalize status
        if status and status.lower() == "all":
            status = None

        result = get_attendance_requests(
            emp_id=emp_id,
            status=status,
            limit=limit,
            offset=offset
        )

        # Format response explicitly
        formatted_requests = []
        for req in result["requests"]:
            formatted_requests.append({
                "id": req.get("id"),
                "emp_id": req.get("emp_id"),
                "request_type": req.get("request_type"),
                "date": req.get("date"),
                "in_time": req.get("in_time"),
                "out_time": req.get("out_time"),
                "reason": req.get("reason"),   # None if missing
                "status": req.get("status")
            })

        return {
            "success": True,
            "total": result["total"],
            "requests": formatted_requests,
            "page": offset // limit + 1,
            "per_page": limit
        }

    except Exception as e:
        logging.error(f"❌ Error fetching attendance requests: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch attendance requests: {str(e)}"
        )


@app.get("/api/attendance-request/all")
async def list_all_requests(
    status: Optional[str] = Query(
        None,
        description="pending / approved / rejected / all"
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get ALL attendance requests for ALL users.

    - No authorization
    - status=all returns everything
    - Pagination supported
    """
    try:
        # Normalize status
        if status and status.lower() == "all":
            status = None

        result = get_attendance_requests(
            emp_id=None,   # 🔥 No user filter
            status=status,
            limit=limit,
            offset=offset
        )

        requests = [
            {
                "id": req["id"],
                "emp_id": req["emp_id"],
                "request_type": req["request_type"],
                "date": req["date"],
                "in_time": req["in_time"],
                "out_time": req["out_time"],
                "reason": req.get("reason"),
                "status": req["status"]
            }
            for req in result["requests"]
        ]

        return {
            "success": True,
            "total": result["total"],
            "requests": requests,
            "page": (offset // limit) + 1,
            "per_page": limit
        }

    except Exception as e:
        logging.error(f"❌ Error fetching all attendance requests: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch attendance requests: {str(e)}"
        )



# @app.post("/api/attendance-request/{request_id}/approve")
# async def approve_request(
#     request_id: int,
#     approval: RequestApproval,
#     current_user: dict = Depends(get_current_user)
# ):
#     """
#     Approve or reject an attendance request.
    
#     Only admins can approve/reject requests.
#     When approved, it logs the attendance automatically.
#     """
#     try:
#         # Only admins can approve
#         if current_user["role"] != "admin":
#             raise HTTPException(
#                 status_code=403,
#                 detail="Only admins can approve/reject requests"
#             )
        
#         # Update request status
#         result = update_request_status(
#             request_id=request_id,
#             status=approval.status,
#             approved_by=current_user["username"],
#             remarks=approval.remarks
#         )
        
#         if not result:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Request {request_id} not found"
#             )
        
#         # If approved, log the attendance
#         if approval.status == "approved":
#             try:
#                 # Get the full request details
#                 req_data = get_attendance_requests(limit=1, offset=0)
#                 request_detail = None
                
#                 for req in req_data["requests"]:
#                     if req["id"] == request_id:
#                         request_detail = req
#                         break
                
#                 if request_detail:
#                     # Log attendance based on request type
#                     if request_detail["request_type"] == "wfh":
#                         # Log WFH attendance
#                         log_attendance(
#                             name=request_detail["name"],
#                             emp_id=request_detail["emp_id"],
#                             date=request_detail["date"],
#                             time=request_detail["in_time"] or "09:00:00",
#                             camera="WFH"
#                         )
                    
#                     elif request_detail["request_type"] == "manual_capture":
#                         # Log manual entry
#                         if request_detail["in_time"]:
#                             log_attendance(
#                                 name=request_detail["name"],
#                                 emp_id=request_detail["emp_id"],
#                                 date=request_detail["date"],
#                                 time=request_detail["in_time"],
#                                 camera="Manual-Entry"
#                             )
                        
#                         # Log manual exit
#                         if request_detail["out_time"]:
#                             log_attendance(
#                                 name=request_detail["name"],
#                                 emp_id=request_detail["emp_id"],
#                                 date=request_detail["date"],
#                                 time=request_detail["out_time"],
#                                 camera="Manual-Exit"
#                             )
                
#             except Exception as log_error:
#                 logging.error(f"⚠️ Failed to log attendance after approval: {log_error}")
        
#         return {
#             "success": True,
#             "message": f"Request {approval.status} successfully",
#             "request_id": request_id,
#             "status": approval.status,
#             "emp_id": result["emp_id"],
#             "name": result["name"],
#             "date": result["date"]
#         }
        
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         logging.error(f"❌ Error approving request: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to process approval: {str(e)}"
#         )


from fastapi import APIRouter, Depends, HTTPException
import logging
from datetime import datetime
from triger import send_approval_notification


@app.post("/api/attendance-request/{request_id}/approve")
async def approve_request(
    request_id: int,
    approval: RequestApproval,
    current_user: dict = Depends(get_current_user)
):
    """
    Approve or reject an attendance request.
    
    - Only admins can approve/reject
    - Logs attendance automatically on approval
    - Sends push notification (approved/rejected)
    """
    try:
        # 🔐 Admin check
        if current_user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Only admins can approve/reject requests"
            )

        # 🔄 Update request status
        result = update_request_status(
            request_id=request_id,
            status=approval.status,
            approved_by=current_user["username"],
            remarks=approval.remarks
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request_id} not found"
            )

        emp_id = result["emp_id"]
        name = result["name"]
        date = result["date"]
        request_type = result["request_type"]

        # 🟢 If approved → log attendance
        if approval.status == "approved":
            try:
                if request_type == "wfh":
                    log_attendance(
                        name=name,
                        emp_id=emp_id,
                        date=date,
                        time=result.get("in_time") or "09:00:00",
                        camera="WFH"
                    )

                elif request_type == "manual_capture":
                    if result.get("in_time"):
                        log_attendance(
                            name=name,
                            emp_id=emp_id,
                            date=date,
                            time=result["in_time"],
                            camera="Manual-Entry"
                        )

                    if result.get("out_time"):
                        log_attendance(
                            name=name,
                            emp_id=emp_id,
                            date=date,
                            time=result["out_time"],
                            camera="Manual-Exit"
                        )

            except Exception as log_error:
                logging.error(
                    f"⚠️ Attendance logging failed for request {request_id}: {log_error}"
                )

        # 🔔 SEND APPROVAL / REJECTION NOTIFICATION
        try:
            send_approval_notification(
                emp_id=emp_id,
                name=name,
                request_type=request_type,
                date=date,
                status=approval.status,
                remarks=approval.remarks
            )
        except Exception as notify_error:
            logging.error(
                f"⚠️ Failed to send approval notification for {emp_id}: {notify_error}"
            )

        # ✅ API Response
        return {
            "success": True,
            "message": f"Request {approval.status} successfully",
            "request_id": request_id,
            "status": approval.status,
            "emp_id": emp_id,
            "name": name,
            "date": date
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        logging.error(f"❌ Error approving request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process request approval"
        )





import asyncio
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
executor = ThreadPoolExecutor(max_workers=4)
MAX_IMAGE_DIMENSION = 1024
THRESHOLD = 0.5
LIVENESS_THRESHOLD = 0.6  # Adjustable threshold for liveness detection


# ============================================
# LIVENESS DETECTION FUNCTIONS
# ============================================

def detect_photo_spoof(frame: np.ndarray) -> Dict[str, Any]:
    """
    Detect if the image is a photo of a photo (spoof attack)
    
    Uses multiple detection methods:
    1. Laplacian Variance (Blur Detection)
    2. Frequency Analysis
    3. Color Distribution Analysis
    4. Edge Density
    
    Returns:
        dict with 'is_live' (bool), 'confidence' (float), and 'metrics' (dict)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ==========================================
        # Method 1: Laplacian Variance (Blur Detection)
        # ==========================================
        # Real faces have more texture/sharpness than photos of photos
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # ==========================================
        # Method 2: Frequency Analysis
        # ==========================================
        # Photos of photos have less high-frequency content
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.sum(
            magnitude_spectrum[
                magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4,
                magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4
            ]
        )
        
        # ==========================================
        # Method 3: Color Distribution Analysis
        # ==========================================
        # Photos of photos often have narrower color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv)
        
        # ==========================================
        # Method 4: Edge Density
        # ==========================================
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # ==========================================
        # Scoring System
        # ==========================================
        scores = []
        
        # Laplacian score (higher = sharper = more likely live)
        if laplacian_var > 100:
            scores.append(1.0)
        elif laplacian_var > 50:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # High frequency energy score
        if high_freq_energy > 1000:
            scores.append(1.0)
        elif high_freq_energy > 500:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Color variance score
        if color_variance > 500:
            scores.append(1.0)
        elif color_variance > 300:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Edge density score
        if edge_density > 0.05:
            scores.append(1.0)
        elif edge_density > 0.03:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Calculate final confidence
        confidence = sum(scores) / len(scores)
        
        # Decision based on threshold
        is_live = confidence >= LIVENESS_THRESHOLD
        
        # Determine specific failure reason
        failure_reasons = []
        if laplacian_var <= 50:
            failure_reasons.append("Image too blurry")
        if high_freq_energy <= 500:
            failure_reasons.append("Low frequency content")
        if color_variance <= 300:
            failure_reasons.append("Limited color range")
        if edge_density <= 0.03:
            failure_reasons.append("Insufficient edge detail")
        
        logging.info(
            f"🔍 Liveness Detection - "
            f"Laplacian: {laplacian_var:.2f}, "
            f"Freq Energy: {high_freq_energy:.2f}, "
            f"Color Var: {color_variance:.2f}, "
            f"Edge Density: {edge_density:.4f}, "
            f"Confidence: {confidence:.2f}, "
            f"Is Live: {is_live}"
        )
        
        return {
            "is_live": is_live,
            "confidence": round(confidence, 3),
            "threshold": LIVENESS_THRESHOLD,
            "metrics": {
                "sharpness": round(laplacian_var, 2),
                "frequency_energy": round(high_freq_energy, 2),
                "color_variance": round(color_variance, 2),
                "edge_density": round(edge_density, 4)
            },
            "failure_reasons": failure_reasons if not is_live else [],
            "interpretation": {
                "sharpness_status": "good" if laplacian_var > 100 else "medium" if laplacian_var > 50 else "poor",
                "frequency_status": "good" if high_freq_energy > 1000 else "medium" if high_freq_energy > 500 else "poor",
                "color_status": "good" if color_variance > 500 else "medium" if color_variance > 300 else "poor",
                "edge_status": "good" if edge_density > 0.05 else "medium" if edge_density > 0.03 else "poor"
            }
        }
        
    except Exception as e:
        logging.error(f"❌ Liveness detection error: {e}")
        # On error, be conservative and reject
        return {
            "is_live": False,
            "confidence": 0.0,
            "threshold": LIVENESS_THRESHOLD,
            "error": str(e),
            "failure_reasons": ["Detection error occurred"]
        }


def check_image_quality(frame: np.ndarray) -> Dict[str, Any]:
    """
    Additional quality checks for live camera input
    
    Checks:
    1. Resolution (minimum requirements)
    2. Brightness distribution (avoid uniform/artificial lighting)
    3. Overall image quality metrics
    
    Returns:
        dict with 'passed' (bool), 'reason' (str), and quality metrics
    """
    try:
        height, width = frame.shape[:2]
        
        # ==========================================
        # Check 1: Resolution
        # ==========================================
        min_resolution = 200
        if height < min_resolution or width < min_resolution:
            return {
                "passed": False,
                "reason": "Resolution too low (minimum 200x200 required)",
                "resolution": f"{width}x{height}",
                "details": {
                    "width": width,
                    "height": height,
                    "min_required": min_resolution
                }
            }
        
        # ==========================================
        # Check 2: Brightness Distribution
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Too uniform might indicate display screen
        if brightness_std < 20:
            return {
                "passed": False,
                "reason": "Image too uniform (possible screen display or poor lighting)",
                "brightness": round(brightness, 2),
                "brightness_std": round(brightness_std, 2),
                "details": {
                    "uniformity_issue": True,
                    "suggestion": "Ensure natural lighting and avoid screen displays"
                }
            }
        
        # ==========================================
        # Check 3: Brightness Range
        # ==========================================
        if brightness < 30:
            return {
                "passed": False,
                "reason": "Image too dark",
                "brightness": round(brightness, 2),
                "details": {
                    "lighting_issue": "too_dark",
                    "suggestion": "Increase lighting or adjust camera settings"
                }
            }
        
        if brightness > 225:
            return {
                "passed": False,
                "reason": "Image too bright (possible overexposure)",
                "brightness": round(brightness, 2),
                "details": {
                    "lighting_issue": "too_bright",
                    "suggestion": "Reduce lighting or adjust camera exposure"
                }
            }
        
        # All checks passed
        return {
            "passed": True,
            "brightness": round(brightness, 2),
            "brightness_std": round(brightness_std, 2),
            "resolution": f"{width}x{height}",
            "quality_score": "good"
        }
        
    except Exception as e:
        logging.error(f"❌ Quality check error: {e}")
        return {
            "passed": False,
            "reason": f"Quality check failed: {str(e)}",
            "error": str(e)
        }


# ============================================
# FACE RECOGNITION WITH LIVENESS
# ============================================

def process_face_recognition_with_liveness(contents: bytes) -> Dict[str, Any]:
    """
    Enhanced face recognition with liveness detection
    
    Process Flow:
    1. Decode image
    2. Liveness detection (anti-spoofing)
    3. Image quality check
    4. Face detection
    5. Face alignment
    6. Embedding extraction
    7. Face matching
    
    Returns:
        dict with success status, match info, or error details
    """
    try:
        # ==========================================
        # Step 1: Decode Image
        # ==========================================
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "invalid_image"}
        
        # ==========================================
        # Step 2: LIVENESS DETECTION (CRITICAL)
        # ==========================================
        liveness_result = detect_photo_spoof(frame)
        
        if not liveness_result["is_live"]:
            logging.warning(
                f"⚠️ Liveness check FAILED - "
                f"Confidence: {liveness_result['confidence']} "
                f"(Threshold: {liveness_result['threshold']})"
            )
            return {
                "error": "liveness_failed",
                "message": "Please use live camera feed, not a photo",
                "liveness": liveness_result
            }
        
        logging.info(
            f"✅ Liveness check PASSED - "
            f"Confidence: {liveness_result['confidence']}"
        )
        
        # ==========================================
        # Step 3: IMAGE QUALITY CHECK
        # ==========================================
        quality_result = check_image_quality(frame)
        
        if not quality_result["passed"]:
            logging.warning(f"⚠️ Quality check failed: {quality_result['reason']}")
            return {
                "error": "quality_failed",
                "message": quality_result["reason"],
                "quality": quality_result
            }
        
        logging.info(f"✅ Quality check passed")
        
        # ==========================================
        # Step 4: Resize Large Images
        # ==========================================
        height, width = frame.shape[:2]
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logging.info(f"📐 Resized image: {width}x{height} → {new_width}x{new_height}")
        
        # ==========================================
        # Step 5: Detect Faces
        # ==========================================
        faces = face_app.get(frame)
        
        if not faces:
            return {"error": "no_face"}
        
        if len(faces) > 1:
            logging.warning(f"⚠️ Multiple faces detected ({len(faces)}), using largest face")
        
        face = faces[0]
        
        # ==========================================
        # Step 6: Align Face for Better Recognition
        # ==========================================
        aligned_face = align_face_improved(frame, face)
        
        if aligned_face is not None:
            aligned_faces = face_app.get(aligned_face)
            if aligned_faces:
                emb = aligned_faces[0].embedding
                logging.info("✅ Using aligned face embedding")
            else:
                emb = face.embedding
                logging.info("⚠️ Alignment detection failed, using original embedding")
        else:
            emb = face.embedding
            logging.info("⚠️ Face alignment failed, using original embedding")
        
        # ==========================================
        # Step 7: Load Cached Embeddings
        # ==========================================
        known_faces, known_embeddings = load_known_face_embeddings_cached()
        
        if not known_embeddings:
            return {"error": "no_employees"}
        
        # ==========================================
        # Step 8: Vectorized Similarity Computation
        # ==========================================
        known_embeddings_array = np.array(known_embeddings)
        
        similarities = np.dot(known_embeddings_array, emb) / (
            np.linalg.norm(known_embeddings_array, axis=1) * np.linalg.norm(emb)
        )
        
        best_face_idx = int(np.argmax(similarities))
        best_face_similarity = float(similarities[best_face_idx])
        
        # ==========================================
        # Step 9: Check Threshold
        # ==========================================
        if best_face_similarity > (1 - THRESHOLD):
            match_info = known_faces[best_face_idx]
            
            logging.info(
                f"✅ Face recognized: {match_info['name']} ({match_info['emp_id']}) - "
                f"Similarity: {best_face_similarity:.4f} - "
                f"Liveness: {liveness_result['confidence']:.3f}"
            )
            
            return {
                "success": True,
                "match": {
                    "emp_id": match_info["emp_id"],
                    "name": match_info["name"],
                    "profile_photo": match_info["profile_photo"],
                    "similarity": round(best_face_similarity, 4)
                },
                "liveness": liveness_result,
                "quality": quality_result
            }
        else:
            logging.info(
                f"❌ No match found - "
                f"Best similarity: {best_face_similarity:.4f} "
                f"(Threshold: {1 - THRESHOLD})"
            )
            return {
                "error": "no_match",
                "similarity": round(best_face_similarity, 4),
                "threshold": round(1 - THRESHOLD, 4)
            }
    
    except Exception as e:
        logging.error(f"❌ Processing error: {str(e)}", exc_info=True)
        return {
            "error": "processing_failed",
            "message": str(e)
        }


# ============================================
# API ENDPOINT
# ============================================

@app.post("/api/find-best-match")
async def recognize_face(
    photo: UploadFile = File(...),
    fcm_token: str = Form(""),
    platform: str = Form("")
):
    """
    🚀 Face Recognition with Liveness Detection
    
    Security Features:
    - Multi-method liveness detection (anti-spoofing)
    - Image quality validation
    - Real-time camera requirement
    - Detailed error reporting
    
    Parameters:
        photo: Image file (JPEG/PNG)
        fcm_token: Firebase Cloud Messaging token (optional)
        platform: Platform identifier (android/ios/web) (optional)
    
    Returns:
        JSON response with match info or detailed error
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # ==========================================
        # Validate File Type
        # ==========================================
        if photo.content_type and not photo.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="❌ File must be an image (JPEG/PNG)"
            )
        
        # ==========================================
        # Validate Platform
        # ==========================================
        if platform and platform.strip() and platform.lower() not in ['android', 'ios', 'web']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="❌ Platform must be 'android', 'ios', or 'web'"
            )
        
        # ==========================================
        # Read Uploaded Image
        # ==========================================
        contents = await photo.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        logging.info(f"📤 Received image: {file_size_mb:.2f} MB")
        
        if file_size_mb > 5:
            logging.warning(
                f"⚠️ Large image ({file_size_mb:.2f} MB). "
                "Consider compressing on mobile app."
            )
        
        # ==========================================
        # Run Face Recognition with Liveness
        # ==========================================
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            process_face_recognition_with_liveness,
            contents
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logging.info(f"⏱️ Processing time: {processing_time:.2f}s")
        
        # ==========================================
        # Handle Errors
        # ==========================================
        if "error" in result:
            
            # Invalid Image
            if result["error"] == "invalid_image":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="❌ Invalid image file"
                )
            
            # 🔥 LIVENESS CHECK FAILED (MOST IMPORTANT)
            elif result["error"] == "liveness_failed":
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "success": False,
                        "error": "liveness_failed",
                        "message": "❌ Liveness check failed. Please use live camera feed, not a photo.",
                        "liveness_details": {
                            "confidence": result["liveness"]["confidence"],
                            "threshold": result["liveness"]["threshold"],
                            "passed": False,
                            "failure_reasons": result["liveness"].get("failure_reasons", []),
                            "metrics": result["liveness"].get("metrics", {}),
                            "interpretation": result["liveness"].get("interpretation", {})
                        },
                        "suggestions": [
                            "Use your device's camera in real-time",
                            "Ensure good natural lighting conditions",
                            "Hold the device steady while capturing",
                            "Avoid using screenshots or printed photos",
                            "Make sure the camera lens is clean"
                        ],
                        "processing_time": round(processing_time, 2)
                    }
                )
            
            # 🔥 QUALITY CHECK FAILED
            elif result["error"] == "quality_failed":
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error": "quality_failed",
                        "message": f"❌ {result['message']}",
                        "quality_details": result.get("quality", {}),
                        "suggestions": [
                            "Ensure adequate lighting",
                            "Use higher resolution camera",
                            "Check camera settings",
                            "Clean camera lens"
                        ],
                        "processing_time": round(processing_time, 2)
                    }
                )
            
            # No Face Detected
            elif result["error"] == "no_face":
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "success": False,
                        "error": "no_face",
                        "message": "❌ No face detected in uploaded photo",
                        "suggestions": [
                            "Ensure your face is clearly visible",
                            "Face the camera directly",
                            "Remove any obstructions",
                            "Improve lighting conditions"
                        ],
                        "best_match": None,
                        "processing_time": round(processing_time, 2)
                    }
                )
            
            # No Employees in System
            elif result["error"] == "no_employees":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="❌ No employee data found in system"
                )
            
            # No Match Found
            elif result["error"] == "no_match":
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "success": False,
                        "error": "no_match",
                        "message": "❌ No match found",
                        "match_details": {
                            "best_similarity": result.get("similarity", 0),
                            "threshold": result.get("threshold", 1 - THRESHOLD),
                            "difference": round(
                                result.get("threshold", 1 - THRESHOLD) - result.get("similarity", 0),
                                4
                            )
                        },
                        "suggestions": [
                            "Ensure you are registered in the system",
                            "Try again with better lighting",
                            "Face the camera directly",
                            "Contact administrator if issue persists"
                        ],
                        "best_match": None,
                        "processing_time": round(processing_time, 2)
                    }
                )
            
            # Processing Failed
            elif result["error"] == "processing_failed":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"❌ Face recognition failed: {result.get('message', 'Unknown error')}"
                )
        
        # ==========================================
        # SUCCESS - Store Device Token & Return
        # ==========================================
        if result.get("success"):
            emp_id = result["match"]["emp_id"]
            token_stored = False
            token_info = None
            
            # Store FCM token if provided
            if fcm_token and fcm_token.strip():
                token_stored = await store_device_token(
                    emp_id=emp_id,
                    fcm_token=fcm_token.strip(),
                    platform=platform.strip() if platform and platform.strip() else None
                )
                
                if token_stored:
                    token_info = {
                        "fcm_token": fcm_token.strip(),
                        "platform": platform.strip() if platform and platform.strip() else "unknown"
                    }
                    logging.info(f"✅ Device token stored for employee {emp_id}")
            
            # 🔥 Get dynamic base URL from baseurl.json
            base_url = get_base_url_from_json()
            
            # Replace URL in response with dynamic base URL
            match_data = result["match"].copy()
            if "profile_photo" in match_data and match_data["profile_photo"]:
                # Extract the path after the domain
                old_url = match_data["profile_photo"]
                # Remove any existing base URL
                if "http://" in old_url or "https://" in old_url:
                    # Extract path after domain
                    path = old_url.split("/", 3)[-1] if "/" in old_url else ""
                    match_data["profile_photo"] = f"{base_url}/{path}"
                else:
                    # If it's already a relative path
                    match_data["profile_photo"] = f"{base_url}/{old_url.lstrip('/')}"
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "best_match": match_data,
                    "security_checks": {
                        "liveness_passed": True,
                        "liveness_confidence": result["liveness"]["confidence"],
                        "quality_passed": True,
                        "quality_score": result["quality"].get("quality_score", "good")
                    },
                    "processing_time": round(processing_time, 2),
                    "device_token_stored": token_stored,
                    "device_token_info": token_info
                }
            )
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logging.error(f"❌ Face recognition error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Face recognition failed: {str(e)}"
        )
    
    finally:
        await photo.close()


# ============================================
# DEBUG ENDPOINT (OPTIONAL)
# ============================================

@app.post("/api/test-liveness")
async def test_liveness_only(photo: UploadFile = File(...)):
    """
    🧪 Test liveness detection without face recognition
    
    Useful for debugging and testing liveness detection parameters.
    
    Returns:
        JSON with liveness and quality check results
    """
    try:
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image"
            )
        
        liveness_result = detect_photo_spoof(frame)
        quality_result = check_image_quality(frame)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "liveness": liveness_result,
                "quality": quality_result,
                "overall_passed": liveness_result["is_live"] and quality_result["passed"],
                "recommendations": []
            }
        )
    
    except Exception as e:
        logging.error(f"❌ Test liveness error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        await photo.close()


# ============================================
# HELPER FUNCTIONS (Assumed to exist in your codebase)
# ============================================

# These functions should be defined elsewhere in your application:
# - align_face_improved(frame, face)
# - load_known_face_embeddings_cached()
# - store_device_token(emp_id, fcm_token, platform)


# ============================================
# HELPER FUNCTION: STORE DEVICE TOKEN
# ============================================
async def store_device_token(emp_id: str, fcm_token: str, platform: Optional[str] = None) -> bool:
    """
    Store or update device token for an employee
    
    Args:
        emp_id: Employee ID
        fcm_token: FCM device token
        platform: Platform identifier (android/ios/web)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if token already exists for this employee
        cursor.execute(
            """
            SELECT id FROM device_tokens 
            WHERE emp_id = %s AND fcm_token = %s
            """,
            (emp_id, fcm_token)
        )
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing token with new platform if provided
            if platform:
                cursor.execute(
                    """
                    UPDATE device_tokens 
                    SET platform = %s, created_at = CURRENT_TIMESTAMP
                    WHERE emp_id = %s AND fcm_token = %s
                    """,
                    (platform, emp_id, fcm_token)
                )
                logging.info(f"📱 Updated device token for {emp_id}")
        else:
            # Insert new token
            cursor.execute(
                """
                INSERT INTO device_tokens (emp_id, fcm_token, platform)
                VALUES (%s, %s, %s)
                """,
                (emp_id, fcm_token, platform)
            )
            logging.info(f"📱 Inserted new device token for {emp_id}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        logging.error(f"❌ Error storing device token: {e}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False



# ============================================
# RELOAD CACHE ENDPOINT (Optional)
# ============================================
@app.post("/api/reload-embeddings")
def reload_embeddings():
    """Admin endpoint to reload embeddings after adding/removing employees"""
    global CACHE_LOADED
    CACHE_LOADED = False
    load_known_face_embeddings_cached()
    return {"status": "success", "message": "Embeddings reloaded"}



# Track running processes
running_processes = {}

@app.post("/api/start")
def start_main_process():
    if "main" not in running_processes:
        p = subprocess.Popen(["python3", "main.py"])
        running_processes["main"] = p
        return {"status": "main started"}
    return {"status": "main already running"}

@app.post("/api/stop")
def stop_all():
    for proc in running_processes.values():
        proc.terminate()
    running_processes.clear()
    return {"status": "stopped"}

@app.get("/api/status")
def get_status():
    return {key: True for key in running_processes}


# 👥 Employee Management
@app.post("/api/add-employee")
async def add_employee(name: str = Form(...), emp_id: str = Form(...), photo: UploadFile = File(...)):
    folder_name = f"{name}_{emp_id}"
    folder_path = os.path.join("known_faces", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    save_path = os.path.join(folder_path, photo.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)

    return {"status": "success", "message": f"Employee {name} ({emp_id}) added."}



@app.get("/api/employees")
def get_employees():
    base_url = "http://10.8.11.183:8000/known_faces"
    employees = []

    if not os.path.exists("known_faces"):
        return {"employees": []}

    for folder in os.listdir("known_faces"):
        parts = folder.split("_")
        if len(parts) == 2:
            name, emp_id = parts
            folder_path = os.path.join("known_faces", folder)

            # Get only valid image files (skip auto_ and .npy files)
            files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and not f.lower().startswith("auto_")
                and not f.lower().endswith(".npy")
            ]

            if files:
                img_url = f"{base_url}/{folder}/{files[0]}"
                employees.append({
                    "name": name,
                    "emp_id": emp_id,
                    "image_url": img_url
                })

    return {"employees": employees}

@app.get("/api/employees-mobile")
def get_employees(current_user: dict = Depends(get_current_user)):
    base_url = "http://10.8.11.183:8000/known_faces"
    employees = []

    if not os.path.exists("known_faces"):
        return {"employees": []}

    for folder in os.listdir("known_faces"):
        parts = folder.split("_")
        if len(parts) != 2:
            continue

        name, emp_id = parts

        # 👤 Employee → sirf apna data
        if current_user["role"] == "employee" and emp_id != current_user["username"]:
            continue

        folder_path = os.path.join("known_faces", folder)

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.lower().startswith("auto_")
            and not f.lower().endswith(".npy")
        ]

        if files:
            employees.append({
                "name": name,
                "emp_id": emp_id,
                "image_url": f"{base_url}/{folder}/{files[0]}"
            })

    return {"employees": employees}



@app.delete("/api/delete-employee/{emp_folder}")
def delete_employee(emp_folder: str):
    path = os.path.join("known_faces", emp_folder)
    if os.path.isdir(path):
        shutil.rmtree(path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Employee not found")


@app.put("/api/edit-employee/{old_folder}")
def edit_employee(old_folder: str, new_name: str = Form(...), new_id: str = Form(...)):
    old_path = os.path.join("known_faces", old_folder)
    new_folder = f"{new_name}_{new_id}"
    new_path = os.path.join("known_faces", new_folder)

    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="Employee not found")

    os.rename(old_path, new_path)
    return {"status": "updated"}




@app.get("/api/known-count")
def known_count():
    path = "known_faces"
    if not os.path.exists(path):
        return {"count": 0}
    return {"count": len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])}



@app.get("/api/logs/all")
def get_logs():
    """
    Fetch ALL logs without pagination.
    Returns:
    {
        "total": int,
        "logs": [...],
        "collective": [...]
    }
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        #  Fetch ALL logs
        cursor.execute(
            """
            SELECT emp_id, name, date, time, camera
            FROM attendance_logs
            ORDER BY date DESC, time DESC
            """
        )
        rows = cursor.fetchall()

        logs = [
            {
                "emp_id": r[0],
                "name": r[1],
                "date": str(r[2]),
                "time": str(r[3]),
                "camera": r[4]
            }
            for r in rows
        ]

        #  Total count
        total = len(logs)

        #  Collective Summary (Grouped Data)
        cursor.execute(
            """
            SELECT 
                emp_id,
                name,
                COUNT(*) AS total_logs,
                MIN(date) AS first_seen,
                MAX(date) AS last_seen,
                ARRAY_AGG(DISTINCT camera) AS cameras
            FROM attendance_logs
            GROUP BY emp_id, name
            ORDER BY name ASC;
            """
        )
        grouped_rows = cursor.fetchall()

        collective = [
            {
                "emp_id": r[0],
                "name": r[1],
                "total_logs": r[2],
                "first_seen": str(r[3]),
                "last_seen": str(r[4]),
                "cameras": r[5],
            }
            for r in grouped_rows
        ]

        return {
            "total": total,
            "logs": logs,
            "collective": collective
        }

    except Exception as e:
        print("Error fetching logs: - fastapi_server.py:621", e)
        return {"total": 0, "logs": [], "collective": []}

    finally:
        if conn:
            conn.close()


@app.get("/api/logs")
def get_logs(
    emp_id: str = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0)
):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Filter by emp_id if provided
        if emp_id:
            cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE emp_id=%s", (emp_id,))
            total = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT emp_id, name, date, time, camera
                FROM attendance_logs
                WHERE emp_id=%s
                ORDER BY date DESC, time DESC
                LIMIT %s OFFSET %s
                """,
                (emp_id, limit, offset)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM attendance_logs")
            total = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT emp_id, name, date, time, camera
                FROM attendance_logs
                ORDER BY date DESC, time DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )

        rows = cursor.fetchall()
        logs = [
            {"emp_id": r[0], "name": r[1], "date": str(r[2]), "time": str(r[3]), "camera": r[4]}
            for r in rows
        ]

        return {"total": total, "logs": logs}

    except Exception as e:
        print(e)
        return {"total": 0, "logs": []}
    finally:
        if conn:
            conn.close()


@app.get("/api/logs/present-today")
def is_present_today(emp_id: str = Query(...)):
    """
    Returns whether the given employee is present today.
    Only requires emp_id.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check if the employee has any log for today
        cursor.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE emp_id=%s AND date = CURRENT_DATE",
            (emp_id,)
        )
        total = cursor.fetchone()[0]

        # If total > 0, employee is present
        return {"emp_id": emp_id, "present": total > 0}

    except Exception as e:
        print("Error fetching attendance:", e)
        return {"emp_id": emp_id, "present": False}
    finally:
        if conn:
            conn.close()
       
       

@app.get("/api/employee-entries-with-photos")
def get_employee_entries_with_photos(
    emp_id: str = Query(..., description="Employee ID is required"),
    type: str = Query("all", enum=["all", "entry", "exit"]),
    date: str = Query(None, description="Optional YYYY-MM-DD")
):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # ---------------------------------------
        # Build SQL Query
        # ---------------------------------------
        query = """
            SELECT emp_id, name, date, time, camera
            FROM attendance_logs
            WHERE emp_id = %s
        """
        params = [emp_id]

        # Optional date filter
        if date:
            query += " AND date = %s"
            params.append(date)

        # entry / exit filter
        if type != "all":
            query += " AND LOWER(camera) = %s"
            params.append(type.lower())

        # SORT: latest date first, earliest time first
        query += " ORDER BY date DESC, time ASC"

        cursor.execute(query, params)
        logs = cursor.fetchall()

        results = []

        # ---------------------------------------
        # Build response + photo path match
        # ---------------------------------------
        for log in logs:
            emp_name = log["name"]
            log_date = str(log["date"])   # YYYY-MM-DD
            log_time = str(log["time"])   # HH:MM:SS
            camera = log["camera"]        # Entry / Exit

            emp_folder = f"{emp_name}_{emp_id}"

            # Folder path (backend file system)
            photo_folder = os.path.join(
                "recognized_photos",
                log_date,
                emp_folder,
                camera
            )

            # Convert time for filename (HH-MM-SS)
            time_for_filename = log_time.replace(":", "-")

            exact_photo_url = None

            if os.path.exists(photo_folder):
                for file in os.listdir(photo_folder):

                    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    # Match HH-MM-SS in filename
                    if time_for_filename in file:
                        base_url = "http://10.8.11.183:8000"
                        exact_photo_url = (
                            f"{base_url}/recognized_photos/"
                            f"{log_date}/{emp_folder}/{camera}/{file}"
                        )
                        break

            results.append({
                "emp_id": emp_id,
                "name": emp_name,
                "date": log_date,
                "time": log_time,
                "camera": camera,
                "photo": exact_photo_url
            })

        return {
            "emp_id": emp_id,
            "filter": type,
            "total": len(results),
            "records": results
        }

    except Exception as e:
        print("ERROR: employee full history: - fastapi_server.py:810", e)
        raise HTTPException(status_code=500, detail="Failed to load employee history")

    finally:
        if conn:
            conn.close()




 


@app.get("/api/logs/aggregate-hours")
def get_aggregated_hours(
    emp_id: str = Query(None),
    date: str = Query(None),
    week_start: str = Query(None),
    month: str = Query(None),
    report_type: str = Query("weekly")
):

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        now = datetime.now()
        today = now.date()

        def fmt(d):
            return d.strftime("%b-%d")

        ranges = []

        # ---------------------------
        # ✅ DAILY
        # ---------------------------
        if report_type == "daily":
            day = datetime.strptime(date, "%Y-%m-%d").date() if date else today
            ranges = [(day, day)]

        # ---------------------------
        # ✅ WEEKLY
        # ---------------------------
        elif report_type == "weekly":
            start = datetime.strptime(week_start, "%Y-%m-%d").date() if week_start else today
            start = start - timedelta(days=start.weekday())
            end = start + timedelta(days=6)
            ranges = [(start, end)]

        # ---------------------------
        # ✅ MONTHLY
        # ---------------------------
        elif report_type == "monthly":
            if month:
                year, mon = map(int, month.split("-"))
            else:
                year, mon = today.year, today.month

            first_day = dt_date(year, mon, 1)
            last_day = dt_date(year, mon, calendar.monthrange(year, mon)[1])
            ranges = [(first_day, last_day)]

        else:
            return {"range_count": 0, "data": []}

        final_output = []

        # ===========================
        # 🔁 MAIN RANGE LOOP
        # ===========================
        for start, end in ranges:

            query = """
                SELECT emp_id, name, DATE(date) as date, time, camera
                FROM attendance_logs
                WHERE DATE(date) BETWEEN %s AND %s
            """
            params = [start, end]

            if emp_id:
                query += " AND emp_id=%s"
                params.append(emp_id)

            cursor.execute(query, tuple(params))
            logs = cursor.fetchall()

            emp_group = defaultdict(list)
            for log in logs:
                emp_group[log["emp_id"]].append(log)

            range_users = []

            # ===========================
            # 🔁 EMPLOYEE LOOP
            # ===========================
            for e_id, emp_logs in emp_group.items():

                name = emp_logs[0]["name"]
                logs_by_date = defaultdict(list)

                for lg in emp_logs:
                    logs_by_date[str(lg["date"])].append(lg)

                total_seconds = 0
                daily_records = []

                # ===========================
                # 🔁 DAILY LOOP
                # ===========================
                for date_key, day_logs in logs_by_date.items():

                    sorted_logs = sorted(day_logs, key=lambda x: x["time"])
                    state = "outside"
                    last_entry = None
                    day_seconds = 0

                    first_entry_time = None
                    last_event_time = None
                    last_event_type = None

                    for lg in sorted_logs:
                        cam = lg["camera"].lower()
                        time_str = str(lg["time"])

                        h, m, s = map(int, time_str.split(":"))
                        t = h * 3600 + m * 60 + s

                        # ✅ FIRST ENTRY
                        if cam == "entry" and not first_entry_time:
                            first_entry_time = time_str

                        # ✅ LAST EVENT
                        last_event_time = time_str
                        last_event_type = cam

                        if cam == "entry" and state == "outside":
                            last_entry = t
                            state = "inside"

                        elif cam == "exit" and state == "inside":
                            day_seconds += t - last_entry
                            last_entry = None
                            state = "outside"

                    # ✅ LIVE RUNNING TIME IF STILL INSIDE
                    if state == "inside" and last_entry and date_key == str(today):
                        now_sec = now.hour * 3600 + now.minute * 60 + now.second
                        day_seconds += max(0, now_sec - last_entry)

                    total_seconds += day_seconds

                    hours = day_seconds // 3600
                    mins = (day_seconds % 3600) // 60

                    # ✅ DAILY STATUS
                    daily_status = "outside"
                    if last_event_type == "entry":
                        daily_status = "inside"

                    daily_records.append({
                        "date": date_key,
                        "working_hours": f"{hours}h {mins}m",
                        "status": daily_status,
                        "first_entry": first_entry_time,
                        "last_event": last_event_time,
                        "last_event_type": last_event_type
                    })

                range_users.append({
                    "emp_id": e_id,
                    "name": name,
                    "total": f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m",
                    "daily_hours": daily_records
                })

            final_output.append({
                "range_start": fmt(start),
                "range_end": fmt(end),
                "employees": range_users
            })

        return {
            "range_count": len(final_output),
            "data": final_output
        }

    except Exception as e:
        print("API ERROR: - fastapi_server.py:999", e)
        return {"error": str(e)}

    finally:
        if conn:
            conn.close()

# from fastapi import Query
# from datetime import datetime, timedelta, date as dt_date
# from collections import defaultdict
# import psycopg2, calendar
# from psycopg2.extras import RealDictCursor

# @app.get("/api/logs/aggregate-hours12")
# def get_aggregated_hours(
#     emp_id: str = Query(None),
#     date: str = Query(None),
#     week_start: str = Query(None),
#     month: str = Query(None),
#     report_type: str = Query("weekly")
# ):
#     conn = None
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)

#         now = datetime.now()
#         today = now.date()

#         def fmt(d):
#             return d.strftime("%b-%d")

#         # ---------------------------
#         # 📅 DATE RANGE
#         # ---------------------------
#         if report_type == "daily":
#             d = datetime.strptime(date, "%Y-%m-%d").date() if date else today
#             ranges = [(d, d)]
#         elif report_type == "weekly":
#             start = datetime.strptime(week_start, "%Y-%m-%d").date() if week_start else today
#             start = start - timedelta(days=start.weekday())
#             end = start + timedelta(days=6)
#             ranges = [(start, end)]
#         elif report_type == "monthly":
#             if month:
#                 year, mon = map(int, month.split("-"))
#             else:
#                 year, mon = today.year, today.month
#             first = dt_date(year, mon, 1)
#             last = dt_date(year, mon, calendar.monthrange(year, mon)[1])
#             ranges = [(first, last)]
#         else:
#             return {"range_count": 0, "data": []}

#         final_output = []

#         # ===========================
#         # 🔁 RANGE LOOP
#         # ===========================
#         for start, end in ranges:

#             # ---------------------------
#             # 🟢 FETCH ATTENDANCE LOGS
#             # ---------------------------
#             query = """
#                 SELECT emp_id, name, DATE(date) as date, time, camera
#                 FROM attendance_logs
#                 WHERE DATE(date) BETWEEN %s AND %s
#             """
#             params = [start, end]
#             if emp_id:
#                 query += " AND emp_id=%s"
#                 params.append(emp_id)
#             cursor.execute(query, tuple(params))
#             logs = cursor.fetchall()

#             emp_logs_map = defaultdict(list)
#             for log in logs:
#                 emp_logs_map[log["emp_id"]].append(log)

#             # ---------------------------
#             # 🟢 FETCH APPROVED WFH REQUESTS
#             # ---------------------------
#             cursor.execute("""
#                 SELECT emp_id, date, in_time, out_time
#                 FROM attendance_requests
#                 WHERE status='approved'
#                   AND request_type='wfh'
#                   AND date BETWEEN %s AND %s
#             """, (start, end))
#             wfh_rows = cursor.fetchall()
#             wfh_map = defaultdict(dict)
#             for r in wfh_rows:
#                 wfh_map[r["emp_id"]][str(r["date"])] = r

#             range_users = []

#             # ===========================
#             # 👤 EMPLOYEE LOOP
#             # ===========================
#             all_emp_ids = set(emp_logs_map.keys()) | set(wfh_map.keys())

#             for e_id in all_emp_ids:
#                 emp_logs = emp_logs_map.get(e_id, [])
#                 name = emp_logs[0]["name"] if emp_logs else f"EMP-{e_id}"
#                 logs_by_date = defaultdict(list)
#                 for lg in emp_logs:
#                     logs_by_date[str(lg["date"])].append(lg)

#                 total_seconds = 0
#                 daily_records = []

#                 # ---------------------------
#                 # 📆 ALL DAYS IN RANGE
#                 # ---------------------------
#                 cur = start
#                 all_dates = []
#                 while cur <= end:
#                     all_dates.append(str(cur))
#                     cur += timedelta(days=1)

#                 # ===========================
#                 # 🔁 DAILY LOOP
#                 # ===========================
#                 for day in all_dates:

#                     day_logs = logs_by_date.get(day, [])
#                     day_seconds = 0
#                     first_entry = None
#                     last_event = None
#                     last_event_type = None
#                     status = "absent"

#                     # ---------------------------
#                     # ✅ WFH FIRST (PRIORITY)
#                     # ---------------------------
#                     if day in wfh_map.get(e_id, {}):
#                         wfh = wfh_map[e_id][day]
#                         if wfh["in_time"] and wfh["out_time"]:
#                             h1, m1, s1 = map(int, str(wfh["in_time"]).split(":"))
#                             h2, m2, s2 = map(int, str(wfh["out_time"]).split(":"))
#                             day_seconds = (h2*3600 + m2*60 + s2) - (h1*3600 + m1*60 + s1)
#                         else:
#                             day_seconds = 8*3600

#                         first_entry = str(wfh["in_time"]) if wfh["in_time"] else "WFH"
#                         last_event = str(wfh["out_time"]) if wfh["out_time"] else "WFH"
#                         last_event_type = "wfh"
#                         status = "wfh"

#                     # ---------------------------
#                     # ✅ NORMAL ATTENDANCE LOGS
#                     # ---------------------------
#                     elif day_logs:
#                         sorted_logs = sorted(day_logs, key=lambda x: x["time"])
#                         state = "outside"
#                         last_entry_sec = None

#                         for lg in sorted_logs:
#                             cam = lg["camera"].lower()
#                             t = lg["time"]
#                             h, m, s = map(int, str(t).split(":"))
#                             sec = h*3600 + m*60 + s

#                             if cam == "entry" and not first_entry:
#                                 first_entry = str(t)

#                             last_event = str(t)
#                             last_event_type = cam

#                             if cam == "entry" and state == "outside":
#                                 last_entry_sec = sec
#                                 state = "inside"
#                             elif cam == "exit" and state == "inside":
#                                 day_seconds += sec - last_entry_sec
#                                 state = "outside"

#                         if state == "inside" and day == str(today):
#                             now_sec = now.hour*3600 + now.minute*60 + now.second
#                             day_seconds += max(0, now_sec - last_entry_sec)

#                         status = "inside" if last_event_type == "entry" else "outside"

#                     # ---------------------------
#                     # ❌ ABSENT (DEFAULT)
#                     # ---------------------------
#                     total_seconds += day_seconds
#                     hours = day_seconds // 3600
#                     mins = (day_seconds % 3600) // 60

#                     daily_records.append({
#                         "date": day,
#                         "working_hours": f"{hours}h {mins}m",
#                         "status": status,
#                         "first_entry": first_entry,
#                         "last_event": last_event,
#                         "last_event_type": last_event_type
#                     })

#                 range_users.append({
#                     "emp_id": e_id,
#                     "name": name,
#                     "total": f"{total_seconds // 3600}h {(total_seconds % 3600)//60}m",
#                     "daily_hours": daily_records
#                 })

#             final_output.append({
#                 "range_start": fmt(start),
#                 "range_end": fmt(end),
#                 "employees": range_users
#             })

#         return {
#             "range_count": len(final_output),
#             "data": final_output
#         }

#     except Exception as e:
#         print("API ERROR:", e)
#         return {"error": str(e)}

#     finally:
#         if conn:
#             conn.close()




# @app.get("/api/logs/aggregate-hours1")
# def get_aggregated_hours(
#     emp_id: str = Query(None),
#     date: str = Query(None),
#     week_start: str = Query(None),
#     month: str = Query(None),
#     report_type: str = Query("weekly")
# ):

#     conn = None
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)

#         now = datetime.now()
#         today = now.date()

#         def fmt(d):
#             return d.strftime("%b-%d")

#         ranges = []

#         # ----------------------------------------------------
#         # 📌 DAILY REPORT
#         # ----------------------------------------------------
#         if report_type == "daily":
#             selected_day = (
#                 datetime.strptime(date, "%Y-%m-%d").date()
#                 if date else today
#             )
#             ranges = [(selected_day, selected_day)]

#         # ----------------------------------------------------
#         # 📌 WEEKLY REPORT (Exclude Today)
#         # ----------------------------------------------------
#         elif report_type == "weekly":
#             start_day = (
#                 datetime.strptime(week_start, "%Y-%m-%d").date()
#                 if week_start else today
#             )

#             # Monday of that week
#             start = start_day - timedelta(days=start_day.weekday())

#             # Sunday of that week
#             end = start + timedelta(days=6)

#             # ❌ Exclude Today
#             if end >= today:
#                 end = today - timedelta(days=1)

#             ranges = [(start, end)]

#         # ----------------------------------------------------
#         # 📌 MONTHLY REPORT (Exclude Today)
#         # ----------------------------------------------------
#         elif report_type == "monthly":
#             if month:
#                 year, mon = map(int, month.split("-"))
#             else:
#                 year, mon = today.year, today.month

#             start = dt_date(year, mon, 1)
#             end = dt_date(year, mon, calendar.monthrange(year, mon)[1])

#             # ❌ Exclude Today
#             if end >= today:
#                 end = today - timedelta(days=1)

#             ranges = [(start, end)]

#         else:
#             return {"range_count": 0, "data": []}

#         final_output = []

#         # ====================================================
#         # 🔁 MAIN RANGE PROCESS
#         # ====================================================
#         for start, end in ranges:

#             query = """
#                 SELECT emp_id, name, DATE(date) AS date, time, camera
#                 FROM attendance_logs
#                 WHERE DATE(date) BETWEEN %s AND %s
#             """
#             params = [start, end]

#             if emp_id:
#                 query += " AND emp_id=%s"
#                 params.append(emp_id)

#             cursor.execute(query, tuple(params))
#             logs = cursor.fetchall()

#             # Group logs by employee
#             emp_group = defaultdict(list)
#             for log in logs:
#                 emp_group[log["emp_id"]].append(log)

#             range_users = []

#             # ====================================================
#             # 🔁 EMPLOYEE PROCESS LOOP
#             # ====================================================
#             for e_id, emp_logs in emp_group.items():

#                 name = emp_logs[0]["name"]
#                 logs_by_date = defaultdict(list)

#                 for lg in emp_logs:
#                     logs_by_date[str(lg["date"])].append(lg)

#                 total_seconds = 0
#                 daily_records = []

#                 # ====================================================
#                 # 🔁 DAILY PROCESS LOOP
#                 # ====================================================
#                 for date_key, day_logs in logs_by_date.items():

#                     sorted_logs = sorted(day_logs, key=lambda x: x["time"])
#                     state = "outside"
#                     last_entry_sec = None
#                     day_seconds = 0

#                     first_entry = None
#                     last_event = None
#                     last_event_type = None

#                     for lg in sorted_logs:
#                         cam = lg["camera"].lower()
#                         time_str = str(lg["time"])
#                         h, m, s = map(int, time_str.split(":"))
#                         t = h * 3600 + m * 60 + s

#                         # First Entry
#                         if cam == "entry" and not first_entry:
#                             first_entry = time_str

#                         # Last Event
#                         last_event = time_str
#                         last_event_type = cam

#                         if cam == "entry" and state == "outside":
#                             last_entry_sec = t
#                             state = "inside"

#                         elif cam == "exit" and state == "inside":
#                             day_seconds += t - last_entry_sec
#                             last_entry_sec = None
#                             state = "outside"

#                     # Live running time (only for today)
#                     if state == "inside" and last_entry_sec and date_key == str(today):
#                         now_sec = now.hour * 3600 + now.minute * 60 + now.second
#                         if now_sec > last_entry_sec:
#                             day_seconds += now_sec - last_entry_sec

#                     total_seconds += day_seconds

#                     hours = day_seconds // 3600
#                     mins = (day_seconds % 3600) // 60

#                     daily_status = "inside" if last_event_type == "entry" else "outside"

#                     daily_records.append({
#                         "date": date_key,
#                         "working_hours": f"{hours}h {mins}m",
#                         "status": daily_status,
#                         "first_entry": first_entry,
#                         "last_event": last_event,
#                         "last_event_type": last_event_type
#                     })

#                 # Employee Summary
#                 range_users.append({
#                     "emp_id": e_id,
#                     "name": name,
#                     "total": f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m",
#                     "daily_hours": sorted(daily_records, key=lambda x: x["date"])
#                 })

#             final_output.append({
#                 "range_start": fmt(start),
#                 "range_end": fmt(end),
#                 "employees": range_users
#             })

#         return {
#             "range_count": len(final_output),
#             "data": final_output
#         }

#     except Exception as e:
#         print("API ERROR: - fastapi_server.py:836", e)
#         return {"error": str(e)}

#     finally:
#         if conn:
#             conn.close()


@app.get("/api/logs/total-hours-entry-exit")
def get_total_hours(
    emp_id: str = Query(None),
    date: str = Query(None),
    week_start: str = Query(None),
    month: str = Query(None),
    report_type: str = Query("weekly")
):
    """
    Returns total hours per employee for daily, weekly, or monthly report.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        today = datetime.now().date()

        def fmt(d):
            return d.strftime("%b-%d")

        # -----------------------------
        # Determine date ranges
        # -----------------------------
        if report_type == "daily":
            selected_day = datetime.strptime(date, "%Y-%m-%d").date() if date else today
            ranges = [(selected_day, selected_day)]

        elif report_type == "weekly":
            start_day = datetime.strptime(week_start, "%Y-%m-%d").date() if week_start else today
            start = start_day - timedelta(days=start_day.weekday())
            end = start + timedelta(days=6)
            ranges = [(start, end)]

        elif report_type == "monthly":
            if month:
                year, mon = map(int, month.split("-"))
            else:
                year, mon = today.year, today.month
            start = dt_date(year, mon, 1)
            end = dt_date(year, mon, calendar.monthrange(year, mon)[1])
            ranges = [(start, end)]
        else:
            return {"range_count": 0, "data": []}

        final_output = []

        # -----------------------------
        # Process each range
        # -----------------------------
        for start, end in ranges:

            query = """
                SELECT emp_id, name, DATE(date) AS date, time, camera
                FROM attendance_logs
                WHERE DATE(date) BETWEEN %s AND %s
            """
            params = [start, end]

            if emp_id:
                query += " AND emp_id=%s"
                params.append(emp_id)

            cursor.execute(query, tuple(params))
            logs = cursor.fetchall()

            emp_group = defaultdict(list)
            for lg in logs:
                emp_group[lg["emp_id"]].append(lg)

            employees_out = []

            # -----------------------------
            # Process each employee
            # -----------------------------
            for e_id, emp_logs in emp_group.items():

                name = emp_logs[0]["name"]
                logs_by_date = defaultdict(list)
                for lg in emp_logs:
                    logs_by_date[str(lg["date"])].append(lg)

                daily_records = []
                total_seconds = 0

                # -----------------------------
                # Process each day
                # -----------------------------
                for date_key, day_logs in logs_by_date.items():
                    sorted_day = sorted(day_logs, key=lambda x: x["time"])

                    first_entry = None
                    last_event = None
                    last_event_type = None

                    for lg in sorted_day:
                        cam = lg["camera"].lower()
                        t_str = str(lg["time"])
                        if cam == "entry" and first_entry is None:
                            first_entry = t_str
                        last_event = t_str
                        last_event_type = cam

                    if not first_entry:
                        continue

                    # Compute seconds
                    h1, m1, s1 = map(int, first_entry.split(":"))
                    first_sec = h1 * 3600 + m1 * 60 + s1

                    if last_event_type == "exit":
                        h2, m2, s2 = map(int, last_event.split(":"))
                        end_sec = h2 * 3600 + m2 * 60 + s2
                    else:
                        if date_key == str(today):
                            now = datetime.now()
                            end_sec = now.hour * 3600 + now.minute * 60 + now.second
                        else:
                            h2, m2, s2 = map(int, last_event.split(":"))
                            end_sec = h2 * 3600 + m2 * 60 + s2

                    diff = max(0, end_sec - first_sec)
                    total_seconds += diff

                    daily_records.append({
                        "date": date_key,
                        "first_entry": first_entry,
                        "last_event": last_event,
                        "last_event_type": last_event_type,
                        "working_hours": f"{diff // 3600}h {(diff % 3600) // 60}m",
                    })

                employees_out.append({
                    "emp_id": e_id,
                    "name": name,
                    "total_hours": f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m",
                    "daily_hours": sorted(daily_records, key=lambda x: x["date"])
                })

            final_output.append({
                "range_start": fmt(start),
                "range_end": fmt(end),
                "employees": employees_out
            })

        return {"range_count": len(final_output), "data": final_output}

    except Exception as e:
        print("API ERROR: - fastapi_server.py:1361", e)
        return {"error": str(e)}

    finally:
        if conn:
            conn.close()

# @app.get("/api/logs/total-hours-entry-exit11")
# def get_total_hours(
#     emp_id: str = Query(None),
#     date: str = Query(None),
#     week_start: str = Query(None),
#     month: str = Query(None),da
#     report_type: str = Query("weekly")
# ):
#     conn = None
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)

#         today = datetime.now().date()

#         def fmt(d):
#             return d.strftime("%b-%d")

#         # -----------------------------
#         # Date ranges
#         # -----------------------------
#         if report_type == "daily":
#             d = datetime.strptime(date, "%Y-%m-%d").date() if date else today
#             ranges = [(d, d)]

#         elif report_type == "weekly":
#             start_day = datetime.strptime(week_start, "%Y-%m-%d").date() if week_start else today
#             start = start_day - timedelta(days=start_day.weekday())
#             end = start + timedelta(days=6)
#             ranges = [(start, end)]

#         elif report_type == "monthly":
#             year, mon = map(int, month.split("-")) if month else (today.year, today.month)
#             start = dt_date(year, mon, 1)
#             end = dt_date(year, mon, calendar.monthrange(year, mon)[1])
#             ranges = [(start, end)]
#         else:
#             return {"range_count": 0, "data": []}

#         final_output = []

#         # -----------------------------
#         # Process ranges
#         # -----------------------------
#         for start, end in ranges:

#             # ---- Attendance logs
#             log_query = """
#                 SELECT emp_id, name, DATE(date) AS date, time, camera
#                 FROM attendance_logs
#                 WHERE DATE(date) BETWEEN %s AND %s
#             """
#             params = [start, end]
#             if emp_id:
#                 log_query += " AND emp_id=%s"
#                 params.append(emp_id)

#             cursor.execute(log_query, params)
#             logs = cursor.fetchall()

#             # ---- Approved requests
#             req_query = """
#                 SELECT emp_id, date, in_time, out_time, request_type
#                 FROM attendance_requests
#                 WHERE status='approved'
#                   AND date BETWEEN %s AND %s
#             """
#             req_params = [start, end]
#             if emp_id:
#                 req_query += " AND emp_id=%s"
#                 req_params.append(emp_id)

#             cursor.execute(req_query, req_params)
#             requests = cursor.fetchall()

#             # Map requests by emp+date
#             req_map = {
#                 (r["emp_id"], str(r["date"])): r
#                 for r in requests
#             }

#             emp_logs = defaultdict(list)
#             for lg in logs:
#                 emp_logs[lg["emp_id"]].append(lg)

#             employees_out = []

#             # -----------------------------
#             # Per employee
#             # -----------------------------
#             for e_id, emp_data in emp_logs.items():

#                 name = emp_data[0]["name"]
#                 logs_by_date = defaultdict(list)
#                 for lg in emp_data:
#                     logs_by_date[str(lg["date"])].append(lg)

#                 daily_records = []
#                 total_seconds = 0

#                 # -----------------------------
#                 # Per day
#                 # -----------------------------
#                 for d, day_logs in logs_by_date.items():
#                     sorted_logs = sorted(day_logs, key=lambda x: x["time"])

#                     first_entry = None
#                     last_event = None
#                     last_type = None

#                     for lg in sorted_logs:
#                         if lg["camera"].lower() == "entry" and not first_entry:
#                             first_entry = str(lg["time"])
#                         last_event = str(lg["time"])
#                         last_type = lg["camera"].lower()

#                     # ---- Override using approved request
#                     req = req_map.get((e_id, d))
#                     status = "outside"

#                     if req:
#                         first_entry = str(req["in_time"])
#                         last_event = str(req["out_time"])
#                         last_type = req["request_type"]
#                         status = req["request_type"]
#                     elif not first_entry:
#                         continue

#                     # ---- Calculate time
#                     h1, m1, s1 = map(int, first_entry.split(":"))
#                     start_sec = h1*3600 + m1*60 + s1

#                     if last_event:
#                         h2, m2, s2 = map(int, last_event.split(":"))
#                         end_sec = h2*3600 + m2*60 + s2
#                     elif d == str(today):
#                         now = datetime.now()
#                         end_sec = now.hour*3600 + now.minute*60 + now.second
#                     else:
#                         end_sec = start_sec

#                     diff = max(0, end_sec - start_sec)
#                     total_seconds += diff

#                     daily_records.append({
#                         "date": d,
#                         "working_hours": f"{diff//3600}h {(diff%3600)//60}m",
#                         "status": status,
#                         "first_entry": first_entry,
#                         "last_event": last_event,
#                         "last_event_type": last_type
#                     })

#                 employees_out.append({
#                     "emp_id": e_id,
#                     "name": name,
#                     "total": f"{total_seconds//3600}h {(total_seconds%3600)//60}m",
#                     "daily_hours": sorted(daily_records, key=lambda x: x["date"])
#                 })

#             final_output.append({
#                 "range_start": fmt(start),
#                 "range_end": fmt(end),
#                 "employees": employees_out
#             })

#         return {"range_count": len(final_output), "data": final_output}

#     except Exception as e:
#         print("API ERROR:", e)
#         return {"error": str(e)}
#     finally:
#         if conn:
#             conn.close()


# @app.get("/api/logs/total-hours-entry-exit")
# def get_total_hours(
#     emp_id: str = Query(None),
#     date: str = Query(None),
#     week_start: str = Query(None),
#     month: str = Query(None),
#     report_type: str = Query("weekly")
# ):
#     """
#     Returns total hours per employee for daily, weekly, or monthly report.
#     """
#     conn = None
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)

#         today = datetime.now().date()

#         def fmt(d):
#             return d.strftime("%b-%d")

#         # -------------------------------------------------
#         # DETERMINE DATE RANGE
#         # -------------------------------------------------
#         if report_type == "daily":
#             selected_day = datetime.strptime(date, "%Y-%m-%d").date() if date else today
#             ranges = [(selected_day, selected_day)]

#         elif report_type == "weekly":
#             start_day = datetime.strptime(week_start, "%Y-%m-%d").date() if week_start else today
#             start = start_day - timedelta(days=start_day.weekday())
#             end = start + timedelta(days=6)
#             ranges = [(start, end)]

#         elif report_type == "monthly":
#             if month:
#                 year, mon = map(int, month.split("-"))
#             else:
#                 year, mon = today.year, today.month
#             start = dt_date(year, mon, 1)
#             end = dt_date(year, mon, calendar.monthrange(year, mon)[1])
#             ranges = [(start, end)]
#         else:
#             return {"range_count": 0, "data": []}

#         final_output = []

#         # -------------------------------------------------
#         # PROCESS EACH RANGE
#         # -------------------------------------------------
#         for start, end in ranges:

#             query = """
#                 SELECT emp_id, name, DATE(date) AS date, time, camera
#                 FROM attendance_logs
#                 WHERE DATE(date) BETWEEN %s AND %s
#             """
#             params = [start, end]

#             if emp_id:
#                 query += " AND emp_id=%s"
#                 params.append(emp_id)

#             cursor.execute(query, tuple(params))
#             logs = cursor.fetchall()

#             # Group logs by emp_id
#             emp_group = defaultdict(list)
#             for lg in logs:
#                 emp_group[lg["emp_id"]].append(lg)

#             employees_out = []

#             # -------------------------------------------------
#             # PROCESS EACH EMPLOYEE
#             # -------------------------------------------------
#             for e_id, emp_logs in emp_group.items():

#                 name = emp_logs[0]["name"]
#                 logs_by_date = defaultdict(list)

#                 # -------------------------------------------------
#                 # Exclude TODAY for Weekly & Monthly
#                 # -------------------------------------------------
#                 for lg in emp_logs:

#                     if report_type in ["weekly", "monthly"]:
#                         if str(lg["date"]) == str(today):
#                             continue

#                     logs_by_date[str(lg["date"])].append(lg)

#                 daily_records = []
#                 total_seconds = 0

#                 # -------------------------------------------------
#                 # PROCESS EACH DAY
#                 # -------------------------------------------------
#                 for date_key, day_logs in logs_by_date.items():

#                     sorted_day = sorted(day_logs, key=lambda x: x["time"])

#                     first_entry = None
#                     last_event = None
#                     last_event_type = None

#                     for lg in sorted_day:
#                         cam = lg["camera"].lower()
#                         t_str = str(lg["time"])

#                         if cam == "entry" and first_entry is None:
#                             first_entry = t_str

#                         last_event = t_str
#                         last_event_type = cam

#                     if not first_entry:
#                         continue

#                     # Convert to seconds
#                     h1, m1, s1 = map(int, first_entry.split(":"))
#                     start_sec = h1 * 3600 + m1 * 60 + s1

#                     if last_event_type == "exit":
#                         h2, m2, s2 = map(int, last_event.split(":"))
#                         end_sec = h2 * 3600 + m2 * 60 + s2
#                     else:
#                         # If last event is not exit, treat it as-is
#                         h2, m2, s2 = map(int, last_event.split(":"))
#                         end_sec = h2 * 3600 + m2 * 60 + s2

#                     diff_sec = max(0, end_sec - start_sec)
#                     total_seconds += diff_sec

#                     daily_records.append({
#                         "date": date_key,
#                         "first_entry": first_entry,
#                         "last_event": last_event,
#                         "last_event_type": last_event_type,
#                         "working_hours": f"{diff_sec // 3600}h {(diff_sec % 3600) // 60}m",
#                     })

#                 employees_out.append({
#                     "emp_id": e_id,
#                     "name": name,
#                     "total_hours": f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m",
#                     "daily_hours": sorted(daily_records, key=lambda x: x["date"])
#                 })

#             final_output.append({
#                 "range_start": fmt(start),
#                 "range_end": fmt(end),
#                 "employees": employees_out
#             })

#         return {"range_count": len(final_output), "data": final_output}

#     except Exception as e:
#         print("API ERROR: - fastapi_server.py:1159", e)
#         return {"error": str(e)}

#     finally:
#         if conn:
#             conn.close()








from datetime import datetime, timedelta

HOLIDAYS = {
    "2025-01-01": "New Year",
    "2025-01-06": "Guru Govind Singh Jayanti",
    "2025-01-26": "Republic Day",
    "2025-03-14": "Holi",
    "2025-04-13": "Vaisakhi",
    "2025-08-15": "Independence Day",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-20": "Diwali",
    "2025-10-21": "Diwali",
    "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas Day"
}


@app.get("/api/calculate-working-hours-full")
def calculate_working_hours_full(
    emp_id: str = Query(None),
    limit: int = Query(200, ge=1),
    offset: int = Query(0, ge=0)
):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Fetch distinct employees
        if emp_id:
            cursor.execute("""
                SELECT DISTINCT emp_id, name
                FROM attendance_logs
                WHERE emp_id = %s
            """, (emp_id,))
        else:
            cursor.execute("""
                SELECT DISTINCT emp_id, name
                FROM attendance_logs
            """)

        employees = cursor.fetchall()
        if not employees:
            return {"total": 0, "working_hours": []}

        # Fetch all logs
        if emp_id:
            cursor.execute("""
                SELECT emp_id, name, date, time, camera
                FROM attendance_logs
                WHERE emp_id = %s
                ORDER BY date DESC, time DESC
            """, (emp_id,))
        else:
            cursor.execute("""
                SELECT emp_id, name, date, time, camera
                FROM attendance_logs
                ORDER BY emp_id, date DESC, time DESC
            """)

        logs = cursor.fetchall()

        # Group logs by employee + date
        logs_by_emp_date = {}
        for log in logs:
            key = (log["emp_id"], str(log["date"]))
            logs_by_emp_date.setdefault(key, []).append(log)

        # Build date range
        if logs:
            first_date = min(str(log["date"]) for log in logs)
        else:
            first_date = datetime.now().strftime("%Y-%m-%d")

        today = datetime.now().strftime("%Y-%m-%d")

        date_list = []
        cur = datetime.strptime(first_date, "%Y-%m-%d")
        end = datetime.strptime(today, "%Y-%m-%d")
        while cur <= end:
            date_list.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)

        date_list = sorted(date_list, reverse=True)

        def is_weekend(date_str):
            return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5

        result = []

        # ---- MAIN LOOP ----
        for emp in employees:
            e_id = emp["emp_id"]
            name = emp["name"]

            for date_str in date_list:
                logs_list = logs_by_emp_date.get((e_id, date_str), [])

                # ---------------------------------------------------
                # 1️⃣ HOLIDAY CHECK — HOLIDAY ALWAYS RETURNS RECORD
                # ---------------------------------------------------
                if date_str in HOLIDAYS:
                    result.append({
                        "emp_id": e_id,
                        "name": name,
                        "date": date_str,
                        "working_hours": HOLIDAYS[date_str],   # Holiday name
                        "entry_count": len([x for x in logs_list if x["camera"].lower() == "entry"]),
                        "exit_count": len([x for x in logs_list if x["camera"].lower() == "exit"]),
                        "first_entry": "-" if not logs_list else min(str(x["time"]) for x in logs_list),
                        "last_exit": "-" if not logs_list else max(str(x["time"]) for x in logs_list),
                        "status": "Holiday"
                    })
                    continue

                # ---------------------------------------------------
                # Weekend skip (same as your existing code)
                # ---------------------------------------------------
                if not logs_list and is_weekend(date_str):
                    continue

                # ---------------------------------------------------
                # ABSENT (no logs)
                # ---------------------------------------------------
                if not logs_list:
                    result.append({
                        "emp_id": e_id,
                        "name": name,
                        "date": date_str,
                        "working_hours": "Absent",
                        "entry_count": 0,
                        "exit_count": 0,
                        "first_entry": "-",
                        "last_exit": "-",
                        "status": "Absent"
                    })
                    continue

                # ---------------------------------------------------
                # NORMAL WORKING HOURS CALCULATION
                # ---------------------------------------------------
                total_seconds = 0
                entry_count = 0
                exit_count = 0
                first_entry = None
                last_exit = None
                last_entry = None
                state = "outside"

                for log in sorted(logs_list, key=lambda x: x["time"]):
                    cam = log["camera"].lower()
                    h, m, s = map(int, str(log["time"]).split(":"))
                    time_sec = h*3600 + m*60 + s

                    if cam == "entry":
                        if state == "outside":
                            last_entry = str(log["time"])
                            state = "inside"
                            entry_count += 1
                            if not first_entry:
                                first_entry = last_entry

                    elif cam == "exit":
                        if state == "inside" and last_entry:
                            eh, em, es = map(int, last_entry.split(":"))
                            total_seconds += max(0, time_sec - (eh*3600 + em*60 + es))
                            last_exit = str(log["time"])
                            exit_count += 1
                            state = "outside"
                            last_entry = None
                        else:
                            exit_count += 1
                            last_exit = str(log["time"])

                # Incomplete session → count until now
                if state == "inside" and last_entry:
                    eh, em, es = map(int, last_entry.split(":"))
                    now = datetime.now().time()
                    now_sec = now.hour*3600 + now.minute*60 + now.second
                    total_seconds += max(0, now_sec - (eh*3600 + em*60 + es))

                if total_seconds == 0:
                    working_hours = "Absent"
                    status = "Absent"
                else:
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    working_hours = f"{hours}h {minutes}m"
                    status = "Present"

                result.append({
                    "emp_id": e_id,
                    "name": name,
                    "date": date_str,
                    "working_hours": working_hours,
                    "entry_count": entry_count,
                    "exit_count": exit_count,
                    "first_entry": first_entry or "-",
                    "last_exit": last_exit or "-",
                    "status": status
                })

        total = len(result)
        paginated = result[offset:offset + limit]

        return {"total": total, "working_hours": paginated}

    except Exception as e:
        print("Error:", e)
        return {"total": 0, "working_hours": []}

    finally:
        if conn:
            conn.close()


def parse_hours_to_float(value: str) -> float:
    """
    Converts '8h 30m' → 8.5
    """
    if not value:
        return 0.0

    h, m = 0, 0
    parts = value.lower().split()

    for p in parts:
        if 'h' in p:
            h = float(p.replace('h', ''))
        elif 'm' in p:
            m = float(p.replace('m', '')) / 60

    return h + m


def get_day_status(total_hours: str, office_hours: str) -> str:
    total = parse_hours_to_float(total_hours)
    office = parse_hours_to_float(office_hours)

    if total < 4 or office < 4:
        return "Half Day"

    if total >= 9 and office >= 8:
        return "Full Day"

    return "Short Day"



# Add this helper function near the top of your file (after imports)
import json
from pathlib import Path

def get_base_url_from_json():
    """Read base URL from baseurl.json file"""
    try:
        baseurl_file = Path("baseurl.json")
        if baseurl_file.exists():
            with open(baseurl_file, 'r') as f:
                data = json.load(f)
                # Check both possible field names
                return data.get('baseurl') or data.get('base_url') or "http://localhost:8000"
        else:
            return "http://localhost:8000"
    except Exception as e:
        print(f"⚠️ Error reading baseurl.json: {e}")
        return "http://localhost:8000"


# Replace your existing API function with this updated version
@app.get("/api/logs/day-status")
def get_day_attendance_status(
    emp_id: str = Query(...),
    date: str = Query(..., description="Reference date YYYY-MM-DD"),
    range: str = Query("day", enum=["day", "week", "month", "year"])
):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.now()

        # ======================
        # DATE RANGE
        # ======================
        if range == "day":
            start_date = end_date = ref_date

        elif range == "week":
            start_date = ref_date - timedelta(days=ref_date.weekday())
            end_date = start_date + timedelta(days=6)

        elif range == "month":
            start_date = ref_date.replace(day=1)
            next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            end_date = next_month - timedelta(days=1)

        else:  # year
            start_date = ref_date.replace(month=1, day=1)
            end_date = ref_date.replace(month=12, day=31)

        # ======================
        # FETCH LOGS
        # ======================
        cursor.execute("""
            SELECT emp_id, name, DATE(date) AS day, time, camera
            FROM attendance_logs
            WHERE emp_id=%s
              AND DATE(date) BETWEEN %s AND %s
            ORDER BY day, time
        """, (emp_id, start_date, end_date))

        logs = cursor.fetchall()

        # ======================
        # GET EMPLOYEE NAME
        # ======================
        emp_name = None
        if logs:
            emp_name = logs[0]["name"]
        else:
            # If no logs, try to get name from known_faces folder
            if os.path.exists("known_faces"):
                for folder in os.listdir("known_faces"):
                    if folder.endswith(f"_{emp_id}"):
                        emp_name = folder.split("_")[0]
                        break

        # Group logs by day
        logs_by_day = {}
        for lg in logs:
            day_str = str(lg["day"])
            logs_by_day.setdefault(day_str, []).append(lg)

        # ======================
        # PROFILE PHOTO (SKIP auto_ & embeddings)
        # ======================
        profile_photo = None
        base_url = get_base_url_from_json()  # 🔥 Read from baseurl.json

        if os.path.exists("known_faces"):
            for folder in os.listdir("known_faces"):
                if folder.endswith(f"_{emp_id}"):
                    folder_path = os.path.join("known_faces", folder)

                    files = [
                        f for f in os.listdir(folder_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        and not f.lower().startswith("auto_")
                        and not f.lower().endswith(".npy")
                    ]

                    if files:
                        profile_photo = f"{base_url}/known_faces/{folder}/{files[0]}"
                    break


        days_response = []
        total_work_seconds = 0
        total_office_seconds = 0

        # ======================
        # PROCESS DAYS (ONLY WITH DATA)
        # ======================
        for day_str, day_logs in logs_by_day.items():
            sorted_logs = sorted(day_logs, key=lambda x: x["time"])

            # ----------------------
            # TOTAL HOURS (FIRST ENTRY → LAST EVENT)
            # ----------------------
            first_entry = None
            last_event = None
            last_event_type = None

            for lg in sorted_logs:
                cam = lg["camera"].lower()
                t = str(lg["time"])

                if cam == "entry" and first_entry is None:
                    first_entry = t

                last_event = t
                last_event_type = cam

            if not first_entry:
                continue

            h1, m1, s1 = map(int, first_entry.split(":"))
            start_sec = h1 * 3600 + m1 * 60 + s1

            if last_event_type == "exit":
                h2, m2, s2 = map(int, last_event.split(":"))
                end_sec = h2 * 3600 + m2 * 60 + s2
            else:
                if day_str == str(today.date()):
                    end_sec = today.hour * 3600 + today.minute * 60 + today.second
                else:
                    h2, m2, s2 = map(int, last_event.split(":"))
                    end_sec = h2 * 3600 + m2 * 60 + s2

            work_seconds = max(0, end_sec - start_sec)
            total_work_seconds += work_seconds
            total_hour = f"{work_seconds//3600}h {(work_seconds%3600)//60}m"

            # ----------------------
            # OFFICE HOURS (ENTRY–EXIT PAIRS)
            # ----------------------
            office_seconds = 0
            state = "outside"
            last_entry_sec = None

            for lg in sorted_logs:
                h, m, s = map(int, str(lg["time"]).split(":"))
                sec = h * 3600 + m * 60 + s

                if lg["camera"].lower() == "entry" and state == "outside":
                    last_entry_sec = sec
                    state = "inside"

                elif lg["camera"].lower() == "exit" and state == "inside":
                    office_seconds += sec - last_entry_sec
                    state = "outside"

            if state == "inside" and day_str == str(today.date()):
                now_sec = today.hour * 3600 + today.minute * 60 + today.second
                office_seconds += now_sec - last_entry_sec

            total_office_seconds += office_seconds
            office_hour = f"{office_seconds//3600}h {(office_seconds%3600)//60}m"

            status = get_day_status(total_hour, office_hour)

            # ----------------------
            # ENTRY / EXIT PHOTOS
            # ----------------------
            entries_exits = []

            if range == "day":
                selected_logs = sorted_logs
            else:
                first_entry_log = None
                last_exit_log = None
                last_log = sorted_logs[-1]

                for lg in sorted_logs:
                    if lg["camera"].lower() == "entry" and first_entry_log is None:
                        first_entry_log = lg
                    if lg["camera"].lower() == "exit":
                        last_exit_log = lg

                selected_logs = []
                if first_entry_log:
                    selected_logs.append(first_entry_log)
                if last_exit_log and last_exit_log != first_entry_log:
                    selected_logs.append(last_exit_log)
                elif last_log != first_entry_log:
                    selected_logs.append(last_log)

            for lg in selected_logs:
                time_str = str(lg["time"])
                camera = lg["camera"]
                emp_folder = f"{lg['name']}_{emp_id}"
                time_key = time_str.replace(":", "-")

                photo_folder = os.path.join(
                    "recognized_photos",
                    day_str,
                    emp_folder,
                    camera
                )

                photo_url = None
                if os.path.exists(photo_folder):
                    for f in os.listdir(photo_folder):
                        if time_key in f:
                            photo_url = (
                                f"{base_url}/recognized_photos/"
                                f"{day_str}/{emp_folder}/{camera}/{f}"
                            )
                            break

                entries_exits.append({
                    "time": time_str,
                    "camera": camera,
                    "photo": photo_url
                })

            days_response.append({
                "date": day_str,
                "present": True,
                "total_hour": total_hour,
                "office_hour": office_hour,
                "status": status,
                "entries_exits": entries_exits
            })

        # ======================
        # FINAL RESPONSE
        # ======================
        return {
            "emp_id": emp_id,
            "name": emp_name,
            "range": range,
            "from": str(start_date),
            "to": str(end_date),
            "days_count": len(days_response),
            "total_hours": f"{total_work_seconds//3600}h {(total_work_seconds%3600)//60}m",
            "office_hours": f"{total_office_seconds//3600}h {(total_office_seconds%3600)//60}m",
            "profile_photo": profile_photo,
            "days": sorted(days_response, key=lambda x: x["date"])
        }

    except Exception as e:
        print("DAY STATUS ERROR:", e)
        return {"error": str(e)}

    finally:
        if conn:
            conn.close()





# Serve static folders
app.mount("/known_faces", StaticFiles(directory="known_faces"), name="known_faces")
app.mount("/recognized_photos", StaticFiles(directory=os.path.abspath("recognized_photos")), name="recognized_photos")
app.mount("/unscanned_photos", StaticFiles(directory=os.path.abspath("Unscanned")), name="unscanned_photos")


# 🎥 Live stream (Entry & Exit)
def generate_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.get("/entry_stream")
def entry_stream():
    return StreamingResponse(
        generate_stream("rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=1"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/exit_stream")
def exit_stream():
    return StreamingResponse(
        generate_stream("rtsp://moogle:Admin_123@10.8.21.47:554/video/live?channel=1&subtype=0"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


#  Anonymous Image Handling
anonymous_router = APIRouter()



@anonymous_router.get("/api/anonymous-dates")
def get_anonymous_dates():
    base = "Anonymous"
    if not os.path.exists(base):
        return {"dates": []}
    dates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return {"dates": sorted(dates, reverse=True)}


@anonymous_router.get("/api/anonymous-images")
def get_anonymous_images(
    date: str,
    filter: str = Query("all", enum=["all", "entry", "exit"]),
    from_hour: str = Query(None),
    to_hour: str = Query(None),
    vehicle_type: str = Query("all", enum=["all", "car", "truck"]),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(500, ge=1, le=200, description="Number of images per page")
):
    base = os.path.join("Anonymous", date)
    images = []
    folders = []

    #  Choose Entry/Exit folders
    if filter == "all":
        folders = ["Entry", "Exit"]
    elif filter == "entry":
        folders = ["Entry"]
    elif filter == "exit":
        folders = ["Exit"]

    for sub in folders:
        folder = os.path.join(base, sub)
        if os.path.exists(folder):
            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]
            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(folder, f)),
                reverse=True
            )

            for f in files:
                file_path = os.path.join(folder, f)
                mtime = os.path.getmtime(file_path)
                readable_time = datetime.fromtimestamp(mtime)

                #  Apply optional hour range filters
                if from_hour or to_hour:
                    file_time = readable_time.strftime("%H:%M")
                    if from_hour and file_time < from_hour:
                        continue
                    if to_hour and file_time > to_hour:
                        continue

                #  Vehicle type detection based on filename
                lower_f = f.lower()
                if lower_f.startswith("car_"):
                    detected_vehicle = "car"
                elif lower_f.startswith("truck_"):
                    detected_vehicle = "truck"
                else:
                    detected_vehicle = "unknown"

                #  Apply vehicle_type filter
                if vehicle_type != "all" and detected_vehicle != vehicle_type:
                    continue

                images.append({
                    "path": f"/Anonymous/{date}/{sub}/{f}?ts={int(mtime)}",
                    "timestamp": int(mtime),
                    "datetime": readable_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": sub.lower(),
                    "vehicle_type": detected_vehicle,
                })

    #  Sort newest → oldest
    images.sort(key=lambda x: x["timestamp"], reverse=True)

    #  Pagination logic
    total_images = len(images)
    start = (page - 1) * limit
    end = start + limit
    paginated = images[start:end]
    total_pages = (total_images + limit - 1) // limit  # Ceiling division

    return {
        "date": date,
        "page": page,
        "limit": limit,
        "total_images": total_images,
        "total_pages": total_pages,
        "images": paginated
    }



@anonymous_router.get("/Anonymous/{date}/{cam}/{filename}")
def serve_anonymous_image(date: str, cam: str, filename: str):
    if ".." in date or ".." in cam or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    path = os.path.join("Anonymous", date, cam, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@anonymous_router.post("/api/delete-anonymous-image")
def delete_anonymous_image(data: dict = Body(...)):
    path = data.get("path")
    if not path:
        return {"error": "No path provided"}
    try:
        os.remove(path)
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}






    
@anonymous_router.post("/api/convert-anonymous")
def convert_anonymous(data: dict = Body(...)):
    emp_id = data.get("emp_id")
    emp_name = data.get("name")
    anon_path = data.get("anon_path")
    camera = data.get("camera", "Entry")

    #  Basic validation
    if not (emp_id and emp_name and anon_path):
        return {"error": "emp_id, name, and anon_path are required"}

    emp_folder = f"{emp_name}_{emp_id}"
    emp_folder_path = os.path.join("known_faces", emp_folder)
    if not os.path.exists(emp_folder_path):
        return {"error": f"Employee {emp_name} ({emp_id}) is not registered"}

    abs_path = os.path.join(os.getcwd(), anon_path)
    if not os.path.exists(abs_path):
        return {"error": f"Anonymous image not found at {abs_path}"}

    filename = os.path.basename(abs_path)

    #  Extract date from anon_path (fallback to today)
    date_match = re.search(r"Anonymous[\\/](\d{4}-\d{2}-\d{2})", anon_path)
    date_str = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

    #  Detect vehicle type (for info only, not filename)
    lower_f = filename.lower()
    if lower_f.startswith("car_"):
        vehicle_type = "car"
    elif lower_f.startswith("truck_"):
        vehicle_type = "truck"
    else:
        vehicle_type = "unknown"

    #  Extract time safely from filename
    name_no_ext = re.sub(r'^(car|truck|unknown)[_-]', '', filename.rsplit('.', 1)[0])
    parts = re.split(r'[-_]', name_no_ext)
    try:
        hour, minute, second = parts[:3]
    except ValueError:
        hour, minute, second = datetime.now().strftime("%H %M %S").split()

    time_for_attendance = f"{hour}:{minute}:{second}"

    #  Destination folder structure
    dest_folder = os.path.join("recognized_photos", date_str, emp_folder, camera)
    os.makedirs(dest_folder, exist_ok=True)

    #  New filename (no vehicle prefix)
    new_filename = f"{date_str}_{hour}-{minute}-{second}.jpg"
    dest_path = os.path.join(dest_folder, new_filename)

    #  Move file
    shutil.move(abs_path, dest_path)

    #  Log attendance
    try:
        log_attendance(emp_name, emp_id, date_str, time_for_attendance, camera)
    except Exception as e:
        print(f" Attendance logging failed: {e} - fastapi_server.py:716")

    #  Generate frontend-accessible URL
    base_url = "http://10.8.11.183:8000"
    image_url = f"{base_url}/recognized_photos/{date_str}/{emp_folder}/{camera}/{new_filename}"

    #  Response
    return {
        "success": True,
        "message": "Image converted successfully",
        "vehicle_type": vehicle_type,
        "image_url": image_url
    }




# Include router
app.include_router(anonymous_router)



#  DAILY SUMMARY
@app.post("/api/save-daily-summary")
async def save_daily_summary_api(request: Request):
    data = await request.json()
    emp_id = data.get("emp_id")
    name = data.get("name")
    date = data.get("date")
    working_hours = data.get("working_hours")
    entry_count = data.get("entry_count")
    exit_count = data.get("exit_count")
    first_entry = data.get("first_entry")
    last_exit = data.get("last_exit")
    status = data.get("status")

    if not (emp_id and name and date):
        raise HTTPException(status_code=400, detail="Missing required fields")

    save_daily_summary(emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status)
    return {"success": True}


@app.get("/api/daily-summary")
def get_daily_summary_api(date: str):
    return {"summary": get_daily_summary(date)}



@app.get("/api/average-working-hours")
def get_average_working_hours():
    """Calculate average working hours per employee (excluding today)."""
    conn = None
    try:
        #  Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        today = datetime.now().date()

        #  Fetch working_hours strings excluding today
        cursor.execute("""
            SELECT name, emp_id, working_hours
            FROM daily_summary
            WHERE date < %s
        """, (today,))

        rows = cursor.fetchall()
        cursor.close()

        #  Convert working_hours ("8h 30m") → total minutes
        time_map = {}

        for name, emp_id, wh in rows:
            if not wh:
                continue
            try:
                parts = wh.split()
                hours = int(parts[0].replace("h", "")) if "h" in parts[0] else 0
                minutes = int(parts[1].replace("m", "")) if len(parts) > 1 else 0
                total_minutes = hours * 60 + minutes
            except Exception:
                continue

            if emp_id not in time_map:
                time_map[emp_id] = {"name": name, "total": 0, "count": 0}

            time_map[emp_id]["total"] += total_minutes
            time_map[emp_id]["count"] += 1

        #  Compute averages
        results = []
        for emp_id, info in time_map.items():
            if info["count"] == 0:
                continue

            avg_minutes = info["total"] / info["count"]
            hours = int(avg_minutes // 60)
            minutes = int(avg_minutes % 60)

            results.append({
                "name": info["name"],
                "emp_id": emp_id,
                "avg_hours": f"{hours}h {minutes}m"
            })

        return {"averages": results}

    except Exception as e:
        print(" Error computing averages:", e)
        return {"averages": []}

    finally:
        if conn:
            conn.close()

@app.get("/api/average-working-hours-employee")
def get_average_working_hours_by_user(emp_id: str):
    """Calculate average working hours for a specific employee (excluding today)."""
    conn = None
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        today = datetime.now().date()

        # Fetch working_hours strings for specific employee excluding today
        cursor.execute("""
            SELECT name, working_hours
            FROM daily_summary
            WHERE emp_id = %s AND date < %s
        """, (emp_id, today))

        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            return {
                "emp_id": emp_id,
                "name": None,
                "avg_hours": "0h 0m",
                "total_days": 0,
                "message": "No data found for this employee"
            }

        # Convert working_hours ("8h 30m") → total minutes
        total_minutes = 0
        count = 0
        employee_name = rows[0][0]

        for name, wh in rows:
            if not wh:
                continue
            try:
                parts = wh.split()
                hours = int(parts[0].replace("h", "")) if "h" in parts[0] else 0
                minutes = int(parts[1].replace("m", "")) if len(parts) > 1 else 0
                total_minutes += hours * 60 + minutes
                count += 1
            except Exception:
                continue

        # Compute average
        if count == 0:
            return {
                "emp_id": emp_id,
                "name": employee_name,
                "avg_hours": "0h 0m",
                "total_days": 0,
                "message": "No valid working hours data"
            }

        avg_minutes = total_minutes / count
        hours = int(avg_minutes // 60)
        minutes = int(avg_minutes % 60)

        return {
            "emp_id": emp_id,
            "name": employee_name,
            "avg_hours": f"{hours}h {minutes}m",
            "total_days": count,
            "total_hours": f"{int(total_minutes // 60)}h {int(total_minutes % 60)}m"
        }

    except Exception as e:
        print("Error computing average for user:", e)
        return {
            "emp_id": emp_id,
            "error": str(e)
        }

    finally:
        if conn:
            conn.close()


#  ROOT
@app.get("/")
def root():
    return {"message": " FastAPI Server Running with PostgreSQL "}

