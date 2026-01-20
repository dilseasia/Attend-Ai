import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Eye, X } from "lucide-react";

export default function Log() {
  const navigate = useNavigate();
  const [logs, setLogs] = useState([]);
  const [employees, setEmployees] = useState([]);
  const [searchName, setSearchName] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchDate, setSearchDate] = useState(
    new Date().toISOString().split("T")[0]
  );
  
  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState(false);

  // Keep track of which dates were already saved
  const savedDates = useRef(new Set());

  // Auto-refresh data every 60s
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 60000);
    return () => clearInterval(interval);
  }, []);

  // Refresh when date changes
  useEffect(() => {
    fetchAllData();
  }, [searchDate]);

  const fetchAllData = async () => {
    await Promise.all([fetchEmployees(), fetchLogs()]);
  };

  // 👥 Fetch employees
  const fetchEmployees = async () => {
    try {
      const res = await axios.get("http://10.8.11.183:8000/api/employees");
      const data = Array.isArray(res.data)
        ? res.data
        : res.data?.employees || [];
      setEmployees(data);
    } catch (err) {
      console.error("Error fetching employees:", err);
      setEmployees([]);
    }
  };

  // 📜 Fetch logs
  const fetchLogs = async () => {
    try {
      const res = await axios.get("http://10.8.11.183:8000/api/logs/all");
      const data = Array.isArray(res.data)
        ? res.data
        : res.data?.logs || [];
      setLogs(data);
    } catch (err) {
      console.error("Error fetching logs:", err);
      setLogs([]);
    }
  };

  // 📷 Fetch photo for specific entry/exit
  const fetchPhoto = async (empId, time, camera) => {
    setModalLoading(true);
    setModalError(false);
    setModalOpen(true);
    setModalImage(null);

    try {
      const res = await axios.get(
        `http://10.8.11.183:8000/api/employee-entries-with-photos`,
        {
          params: {
            emp_id: empId,
            type: camera.toLowerCase(),
            date: searchDate,
          },
        }
      );

      const records = res.data?.records || [];
      
      // Find the exact record matching the time
      const matchingRecord = records.find((record) => record.time === time);

      if (matchingRecord && matchingRecord.photo) {
        setModalImage(matchingRecord.photo);
      } else {
        setModalError(true);
      }
    } catch (err) {
      console.error("Error fetching photo:", err);
      setModalError(true);
    } finally {
      setModalLoading(false);
    }
  };

  // 🕒 Improved working hours + entry/exit calculation
  const calculateWorkingDetails = (empId) => {
    const empLogs = logs.filter(
      (log) =>
        (log.employee_id || log.emp_id) === empId &&
        (log.date === searchDate ||
          (log.timestamp &&
            new Date(log.timestamp).toISOString().split("T")[0] === searchDate))
    );

    if (empLogs.length === 0)
      return {
        workingHours: "-",
        entryCount: 0,
        exitCount: 0,
        firstEntry: "-",
        lastExit: "-",
        firstEntryCamera: null,
        lastExitCamera: null,
      };

    // Sort by time
    const sortedLogs = [...empLogs].sort((a, b) =>
      (a.time || a.timestamp).localeCompare(b.time || b.timestamp)
    );

    let totalSeconds = 0;
    let entryCount = 0;
    let exitCount = 0;
    let firstEntry = null;
    let lastExit = null;
    let firstEntryCamera = null;
    let lastExitCamera = null;

    let lastEntry = null;
    let state = "outside";

    sortedLogs.forEach((log) => {
      const cam = (log.camera || "").toLowerCase();
      const timeVal = log.time || log.timestamp;

      const [h, m, s] = (timeVal || "00:00:00").split(":").map(Number);
      const timeInSec = h * 3600 + m * 60 + (s || 0);

      if (cam === "entry") {
        if (state === "outside") {
          lastEntry = timeVal;
          state = "inside";
          entryCount++;
          if (!firstEntry) {
            firstEntry = timeVal;
            firstEntryCamera = "entry";
          }
        }
      } else if (cam === "exit") {
        if (state === "inside" && lastEntry) {
          const [eh, em, es] = lastEntry.split(":").map(Number);
          const entrySec = eh * 3600 + em * 60 + (es || 0);
          const diff = timeInSec - entrySec;
          if (diff > 0) totalSeconds += diff;

          lastExit = timeVal;
          lastExitCamera = "exit";
          exitCount++;
          state = "outside";
          lastEntry = null;
        }
      }
    });

    // Handle ongoing session
    if (state === "inside" && lastEntry) {
      const [eh, em, es] = lastEntry.split(":").map(Number);
      const entrySec = eh * 3600 + em * 60 + (es || 0);
      const now = new Date();
      const currentSec =
        now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
      const diff = currentSec - entrySec;
      if (diff > 0) totalSeconds += diff;
    }

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);

    return {
      workingHours: `${hours}h ${minutes}m`,
      entryCount,
      exitCount,
      firstEntry: firstEntry || "-",
      lastExit: lastExit || "-",
      firstEntryCamera,
      lastExitCamera,
    };
  };

  // 🧩 Combine employees + logs
  const mergedRows = employees
    .filter((e) =>
      e?.name?.toLowerCase()?.includes(searchName?.toLowerCase())
    )
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((emp) => {
      const empId = emp?.emp_id || emp?.employee_id;
      const matchedLogs = logs.filter(
        (l) =>
          String(l?.emp_id).trim() === String(empId).trim() &&
          String(l?.date).trim() === String(searchDate).trim()
      );

      const status = matchedLogs.length > 0 ? "Present" : "Absent";
      const details = calculateWorkingDetails(empId);

      return {
        name: `${emp?.name} (${empId})`,
        employee_id: empId,
        status,
        ...details,
      };
    })
    .filter((row) =>
      statusFilter === "all"
        ? true
        : row?.status?.toLowerCase() === statusFilter
    );

  // 💾 Auto-save daily summary
  useEffect(() => {
    const saveToDailySummary = async () => {
      try {
        if (savedDates.current.has(searchDate)) {
          return;
        }

        for (const row of mergedRows) {
          await axios.post("http://10.8.11.183:8000/api/save-daily-summary", {
            emp_id: row.employee_id,
            name: row.name.split(" (")[0],
            date: searchDate,
            working_hours: row.workingHours,
            entry_count: row.entryCount,
            exit_count: row.exitCount,
            first_entry: row.firstEntry,
            last_exit: row.lastExit,
            status: row.status,
          });
        }

        savedDates.current.add(searchDate);
        console.log(`✅ Daily summary auto-saved for ${searchDate}`);
      } catch (error) {
        console.error("⚠️ Error auto-saving daily summary:", error);
      }
    };

    if (logs.length > 0 && employees.length > 0) {
      saveToDailySummary();
    }
  }, [logs, employees, searchDate]);

  // 📤 Export CSV
  const handleExport = () => {
    if (!mergedRows || mergedRows.length === 0) {
      alert("No data to export.");
      return;
    }

    const headers = [
      "Name (Emp ID)",
      "Status",
      "Working Hours",
      "Entry Count",
      "Exit Count",
      "First Entry",
      "Last Exit",
    ];

    const rows = mergedRows.map((r) =>
      [
        r.name,
        r.status,
        r.workingHours,
        r.entryCount,
        r.exitCount,
        r.firstEntry,
        r.lastExit,
      ]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`)
        .join(",")
    );

    const csvContent = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `attendance_log_${searchDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // 🖥️ UI
  return (
    <motion.div
      className="min-h-screen px-8 py-6"
      style={{
        background: "linear-gradient(135deg, #e0e7ff 0%, #f7fafc 100%)",
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <h1 className="text-3xl font-bold mb-6 text-indigo-600">
        Daily Attendance Log
      </h1>

      {/* 🔍 Filters */}
      <div className="grid md:grid-cols-5 sm:grid-cols-2 gap-4 mb-8">
        <input
          type="text"
          placeholder="Search by name"
          className="border px-4 py-3 rounded-full bg-white shadow"
          value={searchName}
          onChange={(e) => setSearchName(e.target.value)}
        />
        <input
          type="date"
          className="border px-4 py-3 rounded-full bg-white shadow"
          value={searchDate}
          onChange={(e) => setSearchDate(e.target.value)}
        />
        <select
          className="border px-4 py-3 rounded-full bg-white shadow"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="all">All</option>
          <option value="present">Present</option>
          <option value="absent">Absent</option>
        </select>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-4 py-3 rounded-full shadow hover:bg-green-600"
        >
          ⬇ Export CSV
        </button>
      </div>

      {/* 📊 Table */}
      <div className="overflow-x-auto rounded-2xl shadow-lg">
        <table className="w-full border-collapse bg-white text-sm">
          <thead className="bg-blue-50 text-gray-700">
            <tr>
              <th className="px-4 py-3 border">Name (Emp ID)</th>
              <th className="px-4 py-3 border">Status</th>
              <th className="px-4 py-3 border">Working Hours</th>
              <th className="px-4 py-3 border">Entry Count</th>
              <th className="px-4 py-3 border">Exit Count</th>
              <th className="px-4 py-3 border">First Entry</th>
              <th className="px-4 py-3 border">Last Exit</th>
            </tr>
          </thead>
          <tbody>
            {mergedRows.length > 0 ? (
              mergedRows.map((row, i) => (
                <tr
                  key={i}
                  className="text-center hover:bg-blue-50 transition"
                >
                  <td 
                    className="px-4 py-3 border cursor-pointer"
                    onClick={() => navigate(`/report/${row.employee_id}`)}
                  >
                    {row.name}
                  </td>
                  <td 
                    className="px-4 py-3 border cursor-pointer"
                    onClick={() => navigate(`/report/${row.employee_id}`)}
                  >
                    {row.status === "Present" ? (
                      <span className="bg-green-100 text-green-600 px-3 py-1 rounded-full">
                        Present
                      </span>
                    ) : (
                      <span className="bg-red-100 text-red-600 px-3 py-1 rounded-full">
                        Absent
                      </span>
                    )}
                  </td>
                  <td 
                    className="px-4 py-3 border cursor-pointer"
                    onClick={() => navigate(`/report/${row.employee_id}`)}
                  >
                    {row.workingHours}
                  </td>
                  <td 
                    className="px-4 py-3 border cursor-pointer"
                    onClick={() => navigate(`/report/${row.employee_id}`)}
                  >
                    {row.entryCount}
                  </td>
                  <td 
                    className="px-4 py-3 border cursor-pointer"
                    onClick={() => navigate(`/report/${row.employee_id}`)}
                  >
                    {row.exitCount}
                  </td>
                  <td className="px-4 py-3 border">
                    <div className="flex items-center justify-center gap-2">
                      <span>{row.firstEntry}</span>
                      {row.firstEntry !== "-" && row.firstEntryCamera && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            fetchPhoto(
                              row.employee_id,
                              row.firstEntry,
                              row.firstEntryCamera
                            );
                          }}
                          className="text-purple-600 hover:text-purple-800 transition"

                        >
                          <Eye size={18} />
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 border">
                    <div className="flex items-center justify-center gap-2">
                      <span>{row.lastExit}</span>
                      {row.lastExit !== "-" && row.lastExitCamera && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            fetchPhoto(
                              row.employee_id,
                              row.lastExit,
                              row.lastExitCamera
                            );
                          }}
                          className="text-purple-600 hover:text-purple-800 transition"
                        >
                          <Eye size={18} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7" className="text-center text-gray-500 py-4 border">
                  No records found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* 🖼️ Image Modal */}
      {modalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={() => setModalOpen(false)}
        >
          <div
            className="bg-white rounded-lg p-6 max-w-3xl max-h-[90vh] overflow-auto relative"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setModalOpen(false)}
              className="absolute top-4 right-4 text-gray-600 hover:text-gray-900"
            >
              <X size={24} />
            </button>

            {modalLoading && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              </div>
            )}

            {!modalLoading && modalError && (
              <div className="text-center py-12">
                <p className="text-red-600 text-lg">
                  Photo not found or unavailable
                </p>
              </div>
            )}

            {!modalLoading && !modalError && modalImage && (
              <div>
                <h3 className="text-xl font-bold mb-4 text-gray-800">
                  Entry/Exit Photo
                </h3>
                <img
                  src={modalImage}
                  alt="Entry/Exit"
                  className="w-full h-auto rounded-lg"
                  onError={() => setModalError(true)}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}