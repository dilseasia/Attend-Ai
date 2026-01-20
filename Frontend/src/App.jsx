import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Employees from './pages/Employees';
import Attendance from './pages/Attendance';
import Log from './pages/Log';
import LiveCameras from './pages/LiveCameras';
import ProtectedRoute from './components/ProtectedRoute';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Report from './pages/Report'; // Importing Report component
import Anonymous from './pages/Anonymous';
import AttendanceRequest from './pages/attendance_request';

const SIDEBAR_WIDTH = 72; // collapsed width

const Layout = ({ children }) => (
  <div className="flex h-screen bg-gray-50 text-gray-800">
    <Sidebar />
    <div
      className="flex-1 flex flex-col"
      style={{ marginLeft: SIDEBAR_WIDTH }}
    >
      <Header />
      <main className="flex-1 overflow-y-auto p-6">
        {children}
      </main>
    </div>
  </div>
);

export default function App() {
  return (
    <Router>
      <Routes>
        {/* 🔓 Public Route */}
        <Route path="/" element={<Login />} />

        {/* 🔐 Protected Routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Layout>
                <Dashboard />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/employees"
          element={
            <ProtectedRoute>
              <Layout>
                <Employees />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/attendance"
          element={
            <ProtectedRoute>
              <Layout>
                <Attendance />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/log"
          element={
            <ProtectedRoute>
              <Layout>
                <Log />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/live-cameras"
          element={
            <ProtectedRoute>
              <Layout>
                <LiveCameras />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/report/:employeeId" // New route for Report
          element={
            <ProtectedRoute>
              <Layout>
                <Report />
              </Layout>
            </ProtectedRoute>
          }
        />
        <Route path="/anonymous" element={<Layout><Anonymous /></Layout>} />
        <Route
          path="/attendance-request"
          element={
            <ProtectedRoute>
              <Layout>
                <AttendanceRequest />
              </Layout>
            </ProtectedRoute>
          }
        />

      </Routes>
    </Router>
  );
}

[
  { "name": "John", "employee_id": "E001", "date": "2024-06-27", "status": "Present" }
]
