import React, { useState, useEffect } from 'react';
import { FiBell, FiUser, FiLogOut, FiClock } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function Header() {
  const [logs, setLogs] = useState([]);
  const [requests, setRequests] = useState([]);
  const [notifOpen, setNotifOpen] = useState(false);
  const [requestsOpen, setRequestsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const username = localStorage.getItem("username") || "User";
  const navigate = useNavigate();
  
  const [latestSeenLog, setLatestSeenLog] = useState(localStorage.getItem("latestSeenLog") || null);
  const [latestSeenRequest, setLatestSeenRequest] = useState(localStorage.getItem("latestSeenRequest") || null);
  const [hasNewLogs, setHasNewLogs] = useState(false);
  const [hasNewRequests, setHasNewRequests] = useState(false);

  const toggleNotifications = () => {
    const willOpen = !notifOpen;
    setNotifOpen(willOpen);
    setRequestsOpen(false);
    setProfileOpen(false);

    if (willOpen && logs.length > 0) {
      const latestTime = logs[0].date + " " + logs[0].time;
      setLatestSeenLog(latestTime);
      localStorage.setItem("latestSeenLog", latestTime);
      setHasNewLogs(false);
    }
  };

  const toggleRequests = () => {
    const willOpen = !requestsOpen;
    setRequestsOpen(willOpen);
    setNotifOpen(false);
    setProfileOpen(false);

    if (willOpen && requests.length > 0) {
      const latestId = requests[0].id;
      setLatestSeenRequest(latestId);
      localStorage.setItem("latestSeenRequest", latestId);
      setHasNewRequests(false);
    }
  };

  const toggleProfile = () => {
    setProfileOpen(!profileOpen);
    setNotifOpen(false);
    setRequestsOpen(false);
  };

  useEffect(() => {
    fetchLogs();
    fetchRequests();
    const interval = setInterval(() => {
      fetchLogs();
      fetchRequests();
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchLogs = async () => {
    try {
      const res = await axios.get("http://10.8.21.51:8000/api/logs");
      const newLogs = res.data.logs.slice(0, 5);
      setLogs(newLogs);

      if (newLogs.length > 0) {
        const latestTime = newLogs[0].date + " " + newLogs[0].time;
        const storedLatestSeen = localStorage.getItem("latestSeenLog");
        
        if (latestTime !== storedLatestSeen) {
          setHasNewLogs(true);
        } else {
          setHasNewLogs(false);
        }
      }
    } catch (err) {
      console.error("Failed to fetch logs", err);
    }
  };

  const fetchRequests = async () => {
    try {
      const AUTH_TOKEN = 'TOKEN_admin';
      const res = await axios.get("http://10.8.21.51:8000/api/attendance-request/all?status=pending&limit=10", {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`
        }
      });
      const pendingRequests = res.data.requests || [];
      setRequests(pendingRequests);

      if (pendingRequests.length > 0) {
        const latestId = pendingRequests[0].id;
        const storedLatestSeen = localStorage.getItem("latestSeenRequest");
        
        if (latestId !== storedLatestSeen) {
          setHasNewRequests(true);
        } else {
          setHasNewRequests(false);
        }
      }
    } catch (err) {
      console.error("Failed to fetch requests", err);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate("/");
  };

  const getRequestTypeBadge = (type) => {
    const badges = {
      wfh: { label: 'WFH', color: 'bg-indigo-100 text-indigo-700' },
      manual_capture: { label: 'Manual', color: 'bg-purple-100 text-purple-700' }
    };
    const badge = badges[type] || { label: type, color: 'bg-gray-100 text-gray-700' };
    return (
      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge.color}`}>
        {badge.label}
      </span>
    );
  };

  return (
    <motion.div
      className="flex justify-between items-center px-8 py-4 border-b bg-white/80 backdrop-blur-md shadow-lg rounded-b-2xl relative z-50"
      initial={{ opacity: 0, y: -18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        background: 'linear-gradient(90deg, #f0f4ff 60%, #e0e7ff 100%)',
      }}
    >
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-bold tracking-tight text-blue-700 flex items-center gap-2">
          Dashboard
          <span className="ml-2 text-xs font-semibold text-green-700 bg-green-100 px-2 py-0.5 rounded-full shadow animate-pulse border border-green-200">
            ● Live
          </span>
        </h1>
        <span className="ml-4 text-green-600 text-sm hidden md:inline">● All Systems Operational</span>
      </div>
 
      <div className="flex items-center space-x-6 relative">
        {/* 🔔 Attendance Logs Bell */}
        <div className="relative">
          <motion.button
            whileTap={{ scale: 0.9, rotate: -10 }}
            whileHover={{ scale: 1.15 }}
            className="relative p-2 rounded-full bg-blue-50 hover:bg-blue-100 transition shadow"
            onClick={toggleNotifications}
            aria-label="Attendance Logs"
          >
            <FiBell className="text-xl text-blue-700" />
            {hasNewLogs && (
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white animate-pulse"></span>
            )}
          </motion.button>
          <AnimatePresence>
            {notifOpen && (
              <motion.div
                className="absolute right-0 mt-3 w-80 bg-white/95 backdrop-blur-lg border rounded-2xl shadow-2xl overflow-hidden"
                style={{ zIndex: 9999 }}
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.25 }}
              >
                <div className="px-5 py-3 border-b font-semibold text-blue-700 bg-blue-50">
                  🔔 Recent Attendance
                </div>
                <ul className="max-h-64 overflow-auto">
                  {logs.length === 0 ? (
                    <li className="px-5 py-4 text-sm text-gray-500 text-center">No recent entries</li>
                  ) : (
                    logs.map((log, i) => (
                      <li key={i} className="px-5 py-3 text-sm border-b hover:bg-blue-50 transition">
                        <div className="font-medium text-gray-800">{log.name} <span className="text-xs text-gray-400">(ID: {log.employee_id})</span></div>
                        <div className="text-xs text-gray-500">
                          {log.date} @ {log.time} — <span className="capitalize">{log.camera}</span>
                        </div>
                      </li>
                    ))
                  )}
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ⏰ Attendance Requests Bell */}
        <div className="relative">
          <motion.button
            whileTap={{ scale: 0.9, rotate: -10 }}
            whileHover={{ scale: 1.15 }}
            className="relative p-2 rounded-full bg-amber-50 hover:bg-amber-100 transition shadow"
            onClick={toggleRequests}
            aria-label="Attendance Requests"
          >
            <FiClock className="text-xl text-amber-700" />
            {hasNewRequests && requests.length > 0 && (
              <>
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white animate-pulse"></span>
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-bold border-2 border-white">
                  {requests.length}
                </span>
              </>
            )}
            {!hasNewRequests && requests.length > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-amber-500 text-white text-xs rounded-full flex items-center justify-center font-bold border-2 border-white">
                {requests.length}
              </span>
            )}
          </motion.button>
          <AnimatePresence>
            {requestsOpen && (
              <motion.div
                className="absolute right-0 mt-3 w-96 bg-white/95 backdrop-blur-lg border rounded-2xl shadow-2xl overflow-hidden"
                style={{ zIndex: 9999 }}
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.25 }}
              >
                <div className="px-5 py-3 border-b font-semibold text-amber-700 bg-amber-50 flex items-center justify-between">
                  <span>⏰ Pending Requests</span>
                  {requests.length > 0 && (
                    <span className="text-xs bg-amber-200 text-amber-800 px-2 py-0.5 rounded-full font-bold">
                      {requests.length} pending
                    </span>
                  )}
                </div>
                <ul className="max-h-80 overflow-auto">
                  {requests.length === 0 ? (
                    <li className="px-5 py-4 text-sm text-gray-500 text-center">No pending requests</li>
                  ) : (
                    requests.map((request) => (
                      <li key={request.id} className="px-5 py-3 text-sm border-b hover:bg-amber-50 transition cursor-pointer"
                          onClick={() => navigate('/attendance-request')}>
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <div className="font-medium text-gray-800 flex-1">
                            {request.name || request.emp_id}
                          </div>
                          {getRequestTypeBadge(request.request_type)}
                        </div>
                        <div className="text-xs text-gray-500 space-y-0.5">
                          <div>ID: {request.emp_id}</div>
                          <div>Date: {request.date}</div>
                          {request.in_time && <div>In: {request.in_time} {request.out_time && `| Out: ${request.out_time}`}</div>}
                        </div>
                        {request.reason && (
                          <div className="text-xs text-gray-600 mt-1 italic line-clamp-2">
                            "{request.reason}"
                          </div>
                        )}
                      </li>
                    ))
                  )}
                </ul>
                {requests.length > 0 && (
                  <div className="px-5 py-3 border-t bg-amber-50">
                    <button
                      onClick={() => navigate('/attendance-request')}
                      className="w-full text-center text-sm font-medium text-amber-700 hover:text-amber-900 transition"
                    >
                      View All Requests →
                    </button>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* 👤 Profile */}
        <div className="relative">
          <motion.button
            whileTap={{ scale: 0.9, rotate: 8 }}
            whileHover={{ scale: 1.15 }}
            className="p-2 rounded-full bg-blue-50 hover:bg-blue-100 transition shadow"
            onClick={toggleProfile}
            aria-label="Profile"
          >
            <FiUser className="text-xl text-blue-700" />
          </motion.button>
          <AnimatePresence>
            {profileOpen && (
              <motion.div
                className="absolute right-0 mt-3 w-56 bg-white/95 backdrop-blur-lg border rounded-2xl shadow-2xl overflow-hidden"
                style={{ zIndex: 9999 }}
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.25 }}
              >
                <div className="px-5 py-4 border-b flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 text-2xl font-bold shadow">
                    {username[0]?.toUpperCase() || "U"}
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Signed in as</div>
                    <div className="font-semibold text-gray-800">{username}</div>
                  </div>
                </div>
                <button
                  onClick={handleLogout}
                  className="w-full px-5 py-3 text-left hover:bg-blue-50 text-red-600 font-medium flex items-center gap-2 transition"
                >
                  <FiLogOut /> Logout
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}