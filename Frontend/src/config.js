// API Configuration - automatically uses the current hostname
export const API_BASE_URL = typeof window !== 'undefined' 
  ? `http://${window.location.hostname || 'localhost'}:8000`
  : 'http://localhost:8000';
