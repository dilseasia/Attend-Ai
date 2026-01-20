// ============================================
// PART 1 OF 3: IMPORTS, CONFIG & UTILITIES
// ============================================

// React & Hooks
import React, { useState, useEffect, useMemo, useCallback,useRef } from "react";

// Router
import { useParams } from "react-router-dom";

// HTTP Client
import axios from "axios";



// Animations
import { motion, AnimatePresence } from "framer-motion";

// Icons
import {
  FiUser,
  FiCalendar,
  FiRefreshCw,
  FiX,
  FiClock,
  FiChevronLeft,
  FiChevronRight,
  FiLoader,
  FiAlertCircle,
  FiTrendingUp,
  FiActivity,
  FiLogIn,
  FiLogOut,
  FiImage,
} from "react-icons/fi";

// Recharts Components
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  RadialBarChart,
  RadialBar,
  Legend,
  Cell,
} from "recharts";




// Configuration
const BASE_URL = "http://10.8.11.183:8000";
const LOGS_PER_BATCH = 100;
const ITEMS_PER_PAGE = 5;


const HOLIDAYS = {
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
};




// Utility functions
const isWeekend = (dateStr) => {
  if (!dateStr) return false;
  const day = new Date(dateStr + "T00:00:00").getDay();
  return day === 0 || day === 6;
};

const safeDateString = (raw) => (raw ? raw.substring(0, 10) : "");

const formatTimeFromLog = (log) => {
  if (!log) return "00:00:00";
  if (log.time) return log.time;
  if (log.timestamp) return new Date(log.timestamp).toTimeString().split(" ")[0];
  return "00:00:00";
};

const calculateDayWorkingHours = (dayLogs) => {
  if (!dayLogs.length) return { hours: 0, minutes: 0, display: "Absent" };
  
  let totalSeconds = 0;
  let state = "outside";
  let lastEntry = null;
  
  dayLogs.forEach((log) => {
    const cam = (log.camera || "").toLowerCase();
    const [h, m, s] = formatTimeFromLog(log).split(":").map(Number);
    const tsec = h * 3600 + m * 60 + s;
    
    if (cam === "entry") {
      if (state === "outside") {
        state = "inside";
        lastEntry = tsec;
      }
    } else if (cam === "exit") {
      if (state === "inside" && lastEntry != null) {
        totalSeconds += Math.max(0, tsec - lastEntry);
        state = "outside";
        lastEntry = null;
      }
    }
  });
  
  if (state === "inside" && lastEntry != null) {
    const now = new Date();
    const nowSec = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
    totalSeconds += Math.max(0, nowSec - lastEntry);
  }
  
  if (totalSeconds === 0) return { hours: 0, minutes: 0, display: "Absent" };
  
  const hrs = Math.floor(totalSeconds / 3600);
  const mins = Math.floor((totalSeconds % 3600) / 60);
  
  return { hours: hrs, minutes: mins, display: `${hrs}h ${mins}m` };
};

// Loading Spinner
const LoadingSpinner = () => (
  <div className="flex justify-center items-center h-screen bg-gray-50">
    <div className="text-center">
      <FiLoader className="animate-spin text-blue-600 text-5xl mx-auto mb-4" />
      <p className="text-lg font-semibold text-gray-700">Loading your report...</p>
    </div>
  </div>
);

// Error Display
const ErrorDisplay = ({ message }) => (
  <div className="flex flex-col justify-center items-center h-screen bg-gray-50">
    <div className="bg-white p-8 rounded-xl shadow-lg text-center border border-red-100">
      <FiAlertCircle className="text-5xl text-red-500 mx-auto mb-4" />
      <p className="font-semibold text-xl text-gray-800 mb-2">Oops!</p>
      <p className="text-gray-600">{message}</p>
    </div>
  </div>
);

// Employee Header
const EmployeeHeader = ({ employee }) => {
  const [isPresent, setIsPresent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!employee?.emp_id) return;

    const fetchAttendance = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`${BASE_URL}/api/logs/present-today`, {
          params: { emp_id: employee.emp_id },
        });
        setIsPresent(res.data.present);
      } catch (err) {
        setIsPresent(false);
      } finally {
        setLoading(false);
      }
    };
    fetchAttendance();
  }, [employee?.emp_id]);

  const empImage = employee?.image_url
    ? employee.image_url.startsWith("http")
      ? employee.image_url
      : `${BASE_URL}/${employee.image_url}`
    : null;

  return (
    <motion.div
      className="sticky top-0 z-10 mb-6 rounded-2xl shadow-lg border border-gray-200 p-6 transform transition-transform duration-300 hover:-translate-y-1 hover:shadow-2xl"
      style={{
        background: isPresent === null
          ? "linear-gradient(135deg, #f0f0f0 0%, #dcdcdc 100%)"
          : isPresent
          ? "linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%)"
          : "linear-gradient(135deg, #ffe5e5 0%, #ffcccc 100%)",
      }}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center gap-6 md:gap-8">
        <div className="relative">
          <div className="w-20 h-20 md:w-24 md:h-24 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center overflow-hidden shadow-xl">
            {empImage ? (
              <img src={empImage} alt={employee?.name || "Employee"} className="w-full h-full object-cover" />
            ) : (
              <FiUser className="text-4xl md:text-5xl text-white" />
            )}
          </div>
          <div
            className={`absolute -bottom-1 -right-1 w-5 h-5 md:w-6 md:h-6 rounded-full border-3 border-white shadow-lg ${
              isPresent === null
                ? "bg-gray-400 animate-pulse"
                : isPresent
                ? "bg-green-500"
                : "bg-red-500 animate-pulse"
            }`}
          ></div>
        </div>
        <div className="flex-1">
          <h1 className="text-2xl md:text-3xl font-bold text-gray-800">
            {employee?.name || "Employee Report"}
          </h1>
          <p className="text-sm md:text-base text-gray-600 mt-1">
            Employee ID: {employee?.emp_id || employee?.employee_id || "--"}
          </p>
          <span
            className={`inline-block mt-2 px-3 py-1 text-xs md:text-sm font-semibold rounded-full border ${
              isPresent === null
                ? "text-gray-700 bg-gray-200 border-gray-300"
                : isPresent
                ? "text-green-700 bg-green-100 border-green-300"
                : "text-red-700 bg-red-100 border-red-300"
            }`}
          >
            {isPresent === null
              ? "Loading..."
              : isPresent
              ? "Present Today"
              : "Absent Today"}
          </span>
        </div>
      </div>
    </motion.div>
  );
};


// ============================================
// SECTION 1: FILTER CONTROLS WITH DATE SEARCH
// ============================================

// ============================================
// FILTER CONTROLS - DAILY / WEEKLY / MONTHLY ONLY
// ============================================

const FilterControls = ({ reportType, setReportType, onViewAnalytics }) => {
  const options = [
    { type: "daily",   label: "Daily",   gradient: "from-green-500 to-emerald-600" },
    { type: "weekly",  label: "Weekly",  gradient: "from-blue-500 to-indigo-600" },
    { type: "monthly", label: "Monthly", gradient: "from-purple-500 to-fuchsia-600" }
  ];

  return (
    <motion.div
      className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 mb-6"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
          <FiActivity className="text-xl" />
        </div>
        <h3 className="text-[0.9rem] font-bold text-gray-800 uppercase tracking-wider">
          Select Report Type
        </h3>
      </div>

      {/* Buttons */}
      <div className="flex flex-wrap gap-4 items-center">
        {options.map(({ type, label, gradient }) => {
          const isActive = reportType === type;

          return (
            <button
              key={type}
              onClick={() => {
                setReportType(type);
                if (onViewAnalytics) onViewAnalytics(type);
              }}
              className={`
                px-6 py-2.5 rounded-xl font-semibold transition-all flex items-center gap-2
                shadow-sm border
                ${isActive
                  ? `text-white bg-gradient-to-r ${gradient} border-transparent shadow-md scale-[1.03]`
                  : `text-gray-700 bg-gray-100 border-gray-300 hover:bg-gray-200 hover:scale-[1.02]`
                }
              `}
            >
              <FiCalendar className="text-sm" />
              {label}
            </button>
          );
        })}
      </div>
    </motion.div>
  );
};





// ============================================
// PART 2 OF 3: COMPONENTS (Tables, Logs, Analytics Modal)
// Copy this entire section after Part 1
// ============================================

// ============================================
// WORKING HOURS TABLE (FINAL UPDATED VERSION)
// ============================================

// ============================================
// WORKING HOURS TABLE (FINAL + WEEKEND BADGE)
// ============================================

const WorkingHoursTable = ({
  data,
  loading,
  reportType,
  totalHours,
  currentPage,
  onPageChange
}) => {
  const totalPages = Math.ceil(totalHours / ITEMS_PER_PAGE);

  // Helper: detect weekend
  const isWeekendDay = (dateString) => {
    const day = new Date(dateString).getDay(); 
    return day === 0 || day === 6; // Sunday=0, Saturday=6
  };

  return (
    <motion.div
      className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
            <FiTrendingUp className="text-blue-600 text-xl" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-800">
              Working Hours Summary
            </h3>
            <p className="text-xs text-gray-500">
              Track your attendance metrics
            </p>
          </div>
        </div>
        {loading && (
          <FiLoader className="animate-spin text-blue-600 text-2xl" />
        )}
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-gray-200">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                {reportType === "daily" ? "Date" : "Period"}
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                Working Hours
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-200 bg-white">
            {/* Loading */}
            {loading ? (
              <tr>
                <td colSpan="2" className="px-6 py-10 text-center">
                  <FiLoader className="animate-spin mx-auto text-2xl text-blue-500 mb-2" />
                  <p className="text-gray-500 text-sm">Loading data...</p>
                </td>
              </tr>
            ) : data.length === 0 ? (
              /* No Data */
              <tr>
                <td
                  colSpan="2"
                  className="px-6 py-10 text-center text-gray-400"
                >
                  No data available for the selected period
                </td>
              </tr>
            ) : (
              /* Data rows */
              data.map((row, i) => {
                const label = row.date || row.label;

                const isHoliday = row.status === "Holiday";
                const isAbsent = row.status === "Absent";
                const weekendPresent =
                  row.status === "Present" && isWeekendDay(row.date);

                return (
                  <tr
                    key={i}
                    className="transition-colors hover:bg-gray-50"
                  >
                    {/* Date */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-medium text-gray-900">
                        {label}
                      </span>
                    </td>

                    {/* Working Hours Cell */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      {isHoliday ? (
                        // GREEN HOLIDAY TAG
                        <span className="px-3 py-1 text-sm font-semibold text-green-700 bg-green-50 rounded-full border border-green-200">
                          🎉 {row.working_hours}
                        </span>
                      ) : isAbsent ? (
                        // RED ABSENT TAG
                        <span className="px-3 py-1 text-sm font-semibold text-red-700 bg-red-50 rounded-full border border-red-200">
                          Absent
                        </span>
                      ) : weekendPresent ? (
                        // YELLOW WEEKEND PRESENT TAG
                        <span className="px-3 py-1 text-sm font-semibold text-amber-700 bg-amber-100 rounded-full border border-amber-300">
                          Weekend Present • {row.working_hours}
                        </span>
                      ) : (
                        // NORMAL HOURS (Plain Text)
                        <span className="font-semibold text-gray-900">
                          {row.working_hours}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-5 pt-4 border-t border-gray-200">
          <button
            disabled={currentPage === 1 || loading}
            onClick={() => onPageChange(currentPage - 1)}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <FiChevronLeft /> Previous
          </button>

          <span className="text-sm text-gray-600">
            Page{" "}
            <span className="font-semibold text-gray-900">
              {currentPage}
            </span>{" "}
            of{" "}
            <span className="font-semibold text-gray-900">
              {totalPages}
            </span>
          </span>

          <button
            disabled={currentPage >= totalPages || loading}
            onClick={() => onPageChange(currentPage + 1)}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Next <FiChevronRight />
          </button>
        </div>
      )}
    </motion.div>
  );
};



// Log Entry Component
const LogEntry = ({ log, employee, onImageClick }) => {
  const getPhotoPath = () => {
    if (!log) return null;
    const rawDate = log.date || log.timestamp || "";
    const date = safeDateString(rawDate);
    const timeVal = formatTimeFromLog(log);
    const timeStr = timeVal.replace(/:/g, "-");
    const camera = (log.camera || "").charAt(0).toUpperCase() + 
                   (log.camera || "").slice(1).toLowerCase();
    const type = (log.type || "").toLowerCase();

    if (type === "car" || type === "truck") {
      return `${BASE_URL}/Anonymous/${date}/${camera}/${type}_${timeStr}.jpg`;
    }
    if (employee) {
      const emp_id = log.emp_id || log.employee_id;
      const name = encodeURIComponent(employee.name || "Unknown");
      return `${BASE_URL}/recognized_photos/${date}/${name}_${emp_id}/${camera}/${date}_${timeStr}.jpg`;
    }
    return null;
  };

  const photoPath = getPhotoPath();
  const isEntry = (log.camera || "").toLowerCase() === "entry";
  const cameraColor = isEntry
    ? "text-green-700 bg-green-50 border-green-200" 
    : "text-red-700 bg-red-50 border-red-200";

  return (
    <tr className="hover:bg-gray-50 transition-colors">
      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
        {formatTimeFromLog(log)}
      </td>
      <td className="px-4 py-3 whitespace-nowrap">
        <span className={`px-2 py-1 text-xs font-semibold rounded-full border ${cameraColor}`}>
          {log.camera || "--"}
        </span>
      </td>
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
        {(log.type || "--").toString()}
      </td>
      <td className="px-4 py-3 whitespace-nowrap">
        {photoPath ? (
          <img
            src={photoPath}
            alt="Log photo"
            className="h-12 w-12 object-cover rounded-lg cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all"
            onClick={() => onImageClick(photoPath)}
          />
        ) : (
          <span className="text-gray-400 text-sm">No photo</span>
        )}
      </td>
    </tr>
  );
};

// Day Logs Card Component
const DayLogsCard = ({ date, logs, employee, expandedDates, onToggleExpand, onImageClick }) => {
  const sortedLogs = logs.sort((a, b) => 
    formatTimeFromLog(a).localeCompare(formatTimeFromLog(b))
  );
  const showAll = expandedDates[date];
  const visibleLogs = showAll ? sortedLogs : sortedLogs.slice(-5);
  const workingHours = calculateDayWorkingHours(sortedLogs);

  return (
    <div className="mb-6 bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      <div className="bg-gray-50 px-6 py-4 flex justify-between items-center border-b border-gray-200">
        <div className="flex items-center gap-3">
          <FiCalendar className="text-blue-600" />
          <div>
            <span className="font-semibold text-gray-800">{date}</span>
            <span className="text-xs text-gray-500 ml-2">({sortedLogs.length} logs)</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <FiClock className="text-gray-500 text-sm" />
          <span className={`font-semibold ${
            workingHours.display === "Absent" ? "text-red-600" : "text-green-600"
          }`}>
            {workingHours.display}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase">Time</th>
              <th className="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase">Camera</th>
              <th className="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase">Type</th>
              <th className="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase">Photo</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {visibleLogs.map((log, i) => (
              <LogEntry 
                key={i} 
                log={log} 
                employee={employee} 
                onImageClick={onImageClick}
              />
            ))}
          </tbody>
        </table>
      </div>

      {sortedLogs.length > 5 && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
          <button
            className="text-sm font-medium text-blue-600 hover:text-blue-700 transition-colors"
            onClick={() => onToggleExpand(date)}
          >
            {showAll ? "Show Less ↑" : `Show All ${sortedLogs.length} Logs ↓`}
          </button>
        </div>
      )}
    </div>
  );
};

// Image Modal Component
const ImageModal = ({ imageUrl, onClose }) => (
  <motion.div
    className="fixed inset-0 bg-black/80 flex justify-center items-center z-50 p-4"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    onClick={onClose}
  >
    <motion.img
      src={imageUrl}
      alt="Expanded"
      className="max-h-[90vh] max-w-[90vw] rounded-xl shadow-2xl"
      initial={{ scale: 0.9 }}
      animate={{ scale: 1 }}
      exit={{ scale: 0.9 }}
      onClick={(e) => e.stopPropagation()}
    />
    <button
      className="absolute top-6 right-6 w-12 h-12 bg-white rounded-full flex items-center justify-center text-gray-800 hover:bg-gray-100 transition-colors shadow-lg"
      onClick={onClose}
    >
      <FiX className="text-xl" />
    </button>
  </motion.div>
);




const today = new Date();
const currentMonthStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}`;

// Helper to format month display
const formatMonthDisplay = (monthStr) => {
  if (!monthStr) return 'Select month';
  const date = new Date(monthStr + '-01');
  return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
};



// ============================================
// ANALYTICS MODAL - COMPLETELY REDESIGNED
// Copy this entire section to replace your AnalyticsModal component
// ============================================



const AnalyticsModal = ({ isOpen, onClose, reportType, employeeId, employee }) => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [entriesData, setEntriesData] = useState(null);
  const [loadingEntries, setLoadingEntries] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  const [selectedDate, setSelectedDate] = useState(() => new Date().toISOString().slice(0, 10));

  const [weekStart, setWeekStart] = useState(() => {
    const today = new Date();
    const day = (today.getDay() + 6) % 7;
    const monday = new Date(today);
    monday.setDate(today.getDate() - day);
    return monday.toISOString().slice(0, 10);
  });

  
  

  const [photoPreview, setPhotoPreview] = useState({ url: "", time: "", type: "" });

  // FULLSCREEN MODAL REF (for center scroll)
  // const modalRef = useRef(null);
  // FULLSCREEN MODAL REF (for center scroll)
  const modalRef = useRef(null);

  useEffect(() => {
    if (photoPreview.url && modalRef.current) {
      setTimeout(() => {
        modalRef.current.scrollIntoView({
          behavior: "smooth",
          block: "center"
        });
      }, 20);
    }
  }, [photoPreview]);

  

  



  const [selectedMonth, setSelectedMonth] = useState(() => new Date().toISOString().slice(0, 7));

  const todayStr = new Date().toISOString().slice(0, 10);
  const currentMonthStr = new Date().toISOString().slice(0, 7);

  const parseHoursToNumber = (str) => {
      if (!str) return 0;
      const match = str.match(/(\d+)h\s*(\d+)?m?/);
      if (!match) return 0;
      const h = parseInt(match[1] || "0", 10);
      const m = parseInt(match[2] || "0", 10);
      return h + m / 60;
    };

    const getColorByHours = (total_hours, totalHours) => {
    const parseHours = (value) => {
      if (value == null) return 0;
      if (typeof value === "number") return value;

      const str = String(value);

      const hMatch = str.match(/(\d+(?:\.\d+)?)h/);
      const mMatch = str.match(/(\d+)m/);

      const hours = hMatch ? parseFloat(hMatch[1]) : 0;
      const minutes = mMatch ? parseInt(mMatch[1], 10) / 60 : 0;

      return hours + minutes;
    };

    const total = parseHours(total_hours);
    const office = parseHours(totalHours);

    // 🔴 RED
    if (total < 4 || office < 4) {
      return { stroke: "#EF4444", fill: "#FECACA", label: "Half" };
    }

    // 🟢 GREEN (STRICT)
    if (total >= 9 && office >=8) {
      return { stroke: "#10B981", fill: "#D1FAE5", label: "Full" };
    }

    // 🟡 YELLOW
    return { stroke: "#F59E0B", fill: "#FFFBEB", label: "Short" };
  };
  
  const toMinutes = (str = "0h 0m") => {
    const h = parseInt(str.match(/(\d+)h/)?.[1] || 0);
    const m = parseInt(str.match(/(\d+)m/)?.[1] || 0);
    return h * 60 + m;
  };
  const [employeePhotos, setEmployeePhotos] = useState({});

  useEffect(() => {
    const fetchEmployeePhotos = async () => {
      if (!isOpen || (reportType !== 'weekly' && reportType !== 'monthly')) return;
      
      try {
        const params = {
          emp_id: employeeId,
          type: 'all',
        };
        
        // Add date range based on report type
        if (reportType === 'weekly') {
          // Calculate week end date (6 days after start)
          const startDate = new Date(weekStart);
          const endDate = new Date(startDate);
          endDate.setDate(startDate.getDate() + 6);
          
          params.from_date = weekStart;
          params.to_date = endDate.toISOString().slice(0, 10);
        } else if (reportType === 'monthly') {
          params.month = selectedMonth;
        }
        
        const query = new URLSearchParams(params).toString();
        const res = await fetch(`${BASE_URL}/api/employee-entries-with-photos?${query}`);
        const data = await res.json();
        
        // Organize photos by date
        const photosByDate = {};
        if (data.records) {
          data.records.forEach(record => {
            if (!photosByDate[record.date]) {
              photosByDate[record.date] = {
                entries: [],
                exits: []
              };
            }
            if (record.camera.toLowerCase() === 'entry') {
              photosByDate[record.date].entries.push(record);
            } else if (record.camera.toLowerCase() === 'exit') {
              photosByDate[record.date].exits.push(record);
            }
          });
        }
        
        setEmployeePhotos(photosByDate);
      } catch (err) {
        console.error("Error fetching employee photos:", err);
        setEmployeePhotos({});
      }
    };
    
    fetchEmployeePhotos();
  }, [isOpen, reportType, weekStart, selectedMonth, employeeId]);
  

  const calcProgress = (totalStr, officeStr) => {
    const totalMin = toMinutes(totalStr);
    const officeMin = toMinutes(officeStr);

    const OFFICE_TARGET = 8 * 60; // 480
    const TOTAL_TARGET = 9 * 60;  // 540

    // ✅ 100% ONLY if BOTH true
    if (officeMin >= OFFICE_TARGET && totalMin >= TOTAL_TARGET) {
      return 100;
    }

    // ⛔ Prevent showing 100% accidentally
    const percent = (officeMin / OFFICE_TARGET) * 100;
    return Math.min(percent, 99);
  };

  
  


  const formatWeekRange = (startDateStr) => {
    const start = new Date(startDateStr);
    const end = new Date(start);
    end.setDate(start.getDate() + 6);
    const format = (d) =>
      d.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });
    return `${format(start)} - ${format(end)}`;
  };

  // Load analytics data
  const loadAnalytics = useCallback(async () => {
    if (!isOpen) return;
    setLoading(true);
    try {
      const params = { emp_id: employeeId, report_type: reportType };
      if (reportType === "daily") params.date = selectedDate;
      if (reportType === "weekly") params.week_start = weekStart;
      if (reportType === "monthly") params.month = selectedMonth;

      const query = new URLSearchParams(params).toString();
      const res = await fetch(`${BASE_URL}/api/logs/aggregate-hours?${query}`);
      const json = await res.json();

      if (json.data?.length > 0 && json.data[0].employees?.length > 0) {
        const emp = json.data[0].employees[0];
        setAnalyticsData({
          ...emp,
          range_start: json.data[0].range_start,
          range_end: json.data[0].range_end,
          daily_hours: emp.daily_hours ?? [],
          total: emp.total || json.data[0].total || "0h 0m",
          first_entry: emp.first_entry ?? null,
          last_exit: emp.last_exit ?? null,
        });
      } else {
        setAnalyticsData(null);
      }
    } catch (err) {
      console.error("Error loading analytics:", err);
      setAnalyticsData(null);
    } finally {
      setLoading(false);
    }
  }, [isOpen, reportType, selectedDate, weekStart, selectedMonth, employeeId]);

  // Load entries with photos for daily view
  const loadEntries = useCallback(async () => {
    if (!isOpen || reportType !== 'daily') return;
    setLoadingEntries(true);
    try {
      const params = {
        emp_id: employeeId,
        type: 'all',
        date: selectedDate
      };
      const query = new URLSearchParams(params).toString();
      const res = await fetch(`${BASE_URL}/api/employee-entries-with-photos?${query}`);
      const data = await res.json();
      setEntriesData(data);
    } catch (err) {
      console.error("Error loading entries:", err);
      setEntriesData(null);
    } finally {
      setLoadingEntries(false);
    }
  }, [isOpen, reportType, selectedDate, employeeId]);

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  useEffect(() => {
    console.log('cool')
    if (selectedPhoto){
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [selectedPhoto]);

  useEffect(() => {
    if (reportType === 'daily') {
      loadEntries();
    }
  }, [loadEntries, reportType]);

  const prevWeek = () => {
    const d = new Date(weekStart);
    d.setDate(d.getDate() - 7);
    setWeekStart(d.toISOString().slice(0, 10));
  };
  const nextWeek = () => {
    const d = new Date(weekStart);
    d.setDate(d.getDate() + 7);
    setWeekStart(d.toISOString().slice(0, 10));
  };
  const changeDay = (delta) => {
    const d = new Date(selectedDate);
    d.setDate(d.getDate() + delta);
    setSelectedDate(d.toISOString().slice(0, 10));
  };
  const changeMonth = (delta) => {
    const [y, m] = selectedMonth.split("-");
    const d = new Date(parseInt(y, 10), parseInt(m, 10) - 1);
    d.setMonth(d.getMonth() + delta);
    setSelectedMonth(d.toISOString().slice(0, 7));
  };

  // Helper: calculate total hours from daily logs
  const calculateTotalHours = (dailyHours = []) => {
    if (!dailyHours.length) return "0h 0m";

    let totalMinutes = 0;

    dailyHours.forEach(({ first_entry, last_event }) => {
      if (!first_entry || !last_event) return;

      try {
        const [startH, startM] = first_entry.split(":").map(Number);
        const [endH, endM] = last_event.split(":").map(Number);

        const startMinutes = startH * 60 + startM;
        const endMinutes = endH * 60 + endM;

        const diff = endMinutes - startMinutes;
        if (diff > 0) totalMinutes += diff;
      } catch {
        // ignore invalid times
      }
    });

    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;

    return `${hours}h ${minutes}m`;
  };

  // Helper: calculate difference between first & last for a single day
  const getFirstLastDiff = (first, last) => {
    if (!first || !last) return null;

    try {
      const [startH, startM] = first.split(":").map(Number);
      const [endH, endM] = last.split(":").map(Number);

      const startMinutes = startH * 60 + startM;
      const endMinutes = endH * 60 + endM;

      const diff = endMinutes - startMinutes;
      if (diff <= 0) return null;

      const hours = Math.floor(diff / 60);
      const minutes = diff % 60;

      return `${hours}h ${minutes}m`;
    } catch {
      return null;
    }
  };

  const [dailyHours, setDailyHours] = useState([]);

  const [totalHoursResult, setTotalHoursResult] = useState(null);



  useEffect(() => {
    const fetchTotalHours = async () => {
      try {
        const params = { emp_id: employeeId, report_type: reportType };

        if (reportType === "daily") params.date = selectedDate;
        if (reportType === "weekly") params.week_start = weekStart;
        if (reportType === "monthly") params.month = selectedMonth;

        const res = await axios.get(
          "http://10.8.11.183:8000/api/logs/total-hours-entry-exit",
          { params }
        );

        const emp = res.data?.data?.[0]?.employees?.find(
          e => e.emp_id === employeeId
        );

        // ✅ DAILY DATA (KEY FIX)
        const dailyHours1 = emp?.daily_hours || [];
        setDailyHours(dailyHours1);
        

        // ✅ SUMMARY DATA (weekly/monthly card only)
        setTotalHoursResult(emp || null);

        console.log("DAILY STORED →", dailyHours1);

      } catch (err) {
        console.error("Total Hours API Error:", err);
        setDailyHours([]);
        setTotalHoursResult(null);
      }
    };

    fetchTotalHours();
  }, [employeeId, reportType, selectedDate, weekStart, selectedMonth]);

  
  useEffect(() => {
    if (selectedPhoto) {
      // Scroll the modal content container to top
      const modalContent = document.querySelector('.fixed.inset-0.bg-slate-900\\/10');
      if (modalContent) {
        modalContent.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  }, [selectedPhoto]);


  useEffect(() => {
    if (photoPreview?.url) {
      // Wait for modal to mount
      setTimeout(() => {
        window.scrollTo({
          top: 0,
          behavior: "smooth"
        });
      }, 50);
    }
  }, [photoPreview]);

  

  const empImage =
    employee?.image_url
      ? employee.image_url.startsWith("http")
        ? employee.image_url
        : `${BASE_URL}/${employee.image_url}`
      : null;

  const containerVariants = {
    hidden: { opacity: 0, y: 18 },
    visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.04, delayChildren: 0.05 } },
  };
  const itemVariants = { hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } };

  const startDate = analyticsData?.range_start || weekStart || (selectedMonth + "-01");
  const endDate = analyticsData?.range_end || (reportType === "daily" ? selectedDate : selectedMonth + "-28");
  const targetHoursPerDay = 9;
  const totalHours = analyticsData?.total || "0h 0m";
  const totalDays = analyticsData?.daily_hours?.length || 0;

  const chartData = useMemo(() => {
    if (!dailyHours?.length) return [];

    return dailyHours.map((item) => {
      const date = item.date || "N/A";

      const totalHoursStr = item.working_hours || "0h 0m";

      const matchedDay =
        analyticsData?.daily_hours?.find((x) => x.date === date);

      const officeHoursStr =
        matchedDay?.office_hours ||
        matchedDay?.working_hours ||
        "8h 0m";

      const colorResult = getColorByHours(
        totalHoursStr,
        officeHoursStr
      );

      return {
        date,
        hoursNum: parseHoursToNumber(totalHoursStr),
        color: colorResult?.stroke || "#CBD5E1", // 🛡 safety fallback
      };
    });
  }, [dailyHours, analyticsData]);


  

  const CircleProgress = ({ progress, strokeColor }) => {
    const radius = 40;
    const strokeWidth = 8;
    const normalizedRadius = radius - strokeWidth / 2;
    const circumference = normalizedRadius * 2 * Math.PI;
    const strokeDashoffset = circumference - (progress / 100) * circumference;

    return (
      <svg height={radius * 2} width={radius * 2}>
        <circle
          stroke="#e5e7eb"
          fill="transparent"
          strokeWidth={strokeWidth}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          stroke={strokeColor}
          fill="transparent"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          style={{ transition: "stroke-dashoffset 0.9s ease-out" }}
        />
        <text
          x={radius}
          y={radius}
          textAnchor="middle"
          dy="0.3em"
          fontSize="18"
          fontWeight="bold"
          fill="#374151"
        >
          {Math.round(progress)}%
        </text>
      </svg>
    );
  };
  
  
  


  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 bg-slate-900/10 backdrop-blur-sm z-50 flex justify-center items-start overflow-y-auto p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="bg-white w-full max-w-6xl rounded-3xl shadow-2xl overflow-hidden border border-slate-200"
            initial={{ opacity: 0, scale: 0.96, y: 28 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 28 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* HEADER */}
            <div className="sticky top-0 z-20 bg-white/90 backdrop-blur border-b border-slate-200">
              <div className="p-5 flex items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <motion.div
                    className="w-16 h-16 rounded-xl overflow-hidden border border-slate-200 bg-slate-50"
                    initial={{ scale: 0.92, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                  >
                    {empImage ? (
                      <img src={empImage} alt="employee" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <FiUser className="text-slate-400 text-xl" />
                      </div>
                    )}
                  </motion.div>

                  <div>
                    <h3 className="text-lg font-semibold text-slate-900">
                      {reportType.charAt(0).toUpperCase() + reportType.slice(1)} Analytics
                    </h3>
                    <p className="text-xs text-slate-500">{employee?.name}</p>
                    <p className="text-xs text-slate-400 mt-1">
                      {reportType === "daily" ? selectedDate : `${startDate} → ${endDate}`}
                    </p>
                  </div>
                </div>

                {/* PICKERS */}
                <div className="flex items-center gap-3">
                  {reportType === "daily" && (
                    <div className="flex items-center gap-2 bg-white px-3 py-2 rounded-xl border border-slate-200 shadow-sm">
                      <FiCalendar className="text-blue-600" />
                      <input
                        type="date"
                        value={selectedDate}
                        max={todayStr}
                        onChange={(e) => setSelectedDate(e.target.value)}
                        className="bg-transparent outline-none text-sm font-medium"
                      />
                      <div className="ml-2 flex gap-1">
                        <button
                          onClick={() => changeDay(-1)}
                          className="p-1 rounded hover:bg-slate-100"
                          title="Previous day"
                        >
                          <FiChevronLeft />
                        </button>
                        <button
                          onClick={() => changeDay(1)}
                          className="p-1 rounded hover:bg-slate-100"
                          title="Next day"
                        >
                          <FiChevronRight />
                        </button>
                      </div>
                    </div>
                  )}

                  {reportType === "weekly" && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={prevWeek}
                        className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 flex items-center justify-center"
                        title="Previous week"
                      >
                        <FiChevronLeft />
                      </button>
                      <div className="bg-white px-4 py-2 rounded-xl border border-slate-200 shadow-sm flex items-center gap-2">
                        <FiCalendar className="text-blue-600" />
                        <span className="text-sm font-semibold">Week: {formatWeekRange(weekStart)}</span>
                      </div>
                      <button
                        onClick={nextWeek}
                        className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200"
                        title="Next week"
                      >
                        <FiChevronRight />
                      </button>
                    </div>
                  )}

                  {reportType === "monthly" && (
                    <div className="flex items-center gap-3 bg-white px-3 py-2 rounded-xl border border-slate-200 shadow-sm">
                        <FiCalendar className="text-purple-600" />

                        {/* Previous month */}
                        <button
                          onClick={() => {
                            if (!selectedMonth) return;
                            const [year, month] = selectedMonth.split("-");
                            const date = new Date(+year, +month - 1);
                            date.setMonth(date.getMonth() - 1);
                            setSelectedMonth(
                              `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`
                            );
                          }}
                          className="p-1 rounded hover:bg-slate-100"
                        >
                          <FiChevronLeft className="w-4 h-4" />
                        </button>

                        {/* Month Input */}
                        <input
                          type="month"
                          value={selectedMonth || ""}
                          max={`${new Date().getFullYear()}-${String(
                            new Date().getMonth() + 1
                          ).padStart(2, "0")}`}
                          onChange={(e) => setSelectedMonth(e.target.value)}
                          className="text-sm font-medium text-center border border-slate-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-purple-400"
                        />

                        {/* Next month */}
                        <button
                          onClick={() => {
                            if (!selectedMonth) return;

                            const today = new Date();
                            const currentMonthStr = `${today.getFullYear()}-${String(
                              today.getMonth() + 1
                            ).padStart(2, "0")}`;

                            const [year, month] = selectedMonth.split("-");
                            const date = new Date(+year, +month - 1);
                            date.setMonth(date.getMonth() + 1);

                            const newMonth = `${date.getFullYear()}-${String(
                              date.getMonth() + 1
                            ).padStart(2, "0")}`;

                            if (newMonth <= currentMonthStr) {
                              setSelectedMonth(newMonth);
                            }
                          }}
                          className="p-1 rounded hover:bg-slate-100"
                        >
                          <FiChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    
                  )}
                </div>

                <button
                  onClick={() => {
                    onClose();
                    // Trigger a refresh of the parent component's data
                  }}
                  className="p-2 rounded-full hover:bg-slate-100 text-slate-500 transition"
                  aria-label="Close analytics modal"
                >
                  <FiX className="text-xl" />
                </button>

              </div>
            </div>

            {/* BODY */}
            <div className="p-6 space-y-6">
              {loading ? (
                <div className="py-20 flex flex-col items-center gap-3 text-slate-500">
                  <FiLoader className="animate-spin text-blue-500 text-5xl" />
                  <p className="text-sm">Fetching analytics…</p>
                </div>
              ) : !analyticsData ? (
                <div className="py-20 text-center text-slate-500">
                  {reportType === "daily" && isWeekend(selectedDate) ? (
                    <>
                      <div className="text-6xl mb-4">🏖️</div>
                      <p className="text-2xl font-bold text-amber-600 mb-2">Weekend</p>
                      <p className="text-slate-400">No office hours recorded</p>
                    </>
                  ) : reportType === "daily" && HOLIDAYS[selectedDate] ? (
                    <>
                      <div className="text-6xl mb-4">🎊</div>
                      <p className="text-2xl font-bold text-green-600 mb-2">{HOLIDAYS[selectedDate]}</p>
                      <p className="text-slate-400">Holiday - No data recorded</p>
                    </>
                  ) : (
                    <>
                      <FiAlertCircle className="text-5xl text-slate-300 mx-auto mb-4" />
                      <p className="text-lg">Sorry, no data available for the selected {reportType}.</p>
                    </>
                  )}
                </div>
              ) : (

                <motion.div variants={containerVariants} initial="hidden" animate="visible">
                  {/* EMPLOYEE HEADER */}
                  <motion.div
                    variants={itemVariants}
                    className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                  >
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
                      {/* LEFT: EMPLOYEE IMAGE + NAME */}
                      <div className="flex items-center gap-4">
                        <div className="w-20 h-20 rounded-2xl overflow-hidden border border-slate-200 bg-slate-50">
                          {empImage ? (
                            <img src={empImage} alt="employee" className="w-full h-full object-cover" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <FiUser className="text-slate-400 text-3xl" />
                            </div>
                          )}
                        </div>
                        <div>
                          <h4 className="text-xl font-bold text-slate-900">{analyticsData.name}</h4>
                          <p className="text-xs text-slate-500 mt-1">
                            {startDate} → {endDate}
                          </p>
                        </div>
                      </div>

                      {/* RIGHT: METRIC CARDS */}
                      <div className="flex items-center gap-3 flex-wrap">
                          {/* Total Hours */}
                          <div className="px-5 py-3 bg-gradient-to-br from-indigo-50 to-indigo-100 border border-indigo-200 rounded-xl shadow-sm flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-indigo-200 flex items-center justify-center">
                              <span className="text-indigo-700 text-lg">⏱</span>
                            </div>

                            <div>
                              <div className="text-[11px] text-slate-600 uppercase tracking-wide">
                                Total Hours
                              </div>

                              {/* USE ONLY API VALUE */}
                              <div className="text-2xl font-extrabold text-indigo-800 leading-none">
                                {(() => {
                                  if (!totalHoursResult) return "0h 0m";

                                  // DAILY → pick working_hours of selected date
                                  if (reportType === "daily") {
                                    const day = totalHoursResult.daily_hours?.find(
                                      d => d.date === selectedDate
                                    );
                                    return day?.working_hours || "0h 0m";
                                  }

                                  // WEEKLY / MONTHLY → use total_hours from backend
                                  return totalHoursResult.total_hours || "0h 0m";
                                })()}
                              </div>
                              

                              <div className="text-[10px] text-indigo-600 mt-0.5">
                                {reportType === "daily"
                                  ? ""
                                  : ""}
                              </div>
                            </div>
                          </div>



                          {/* Office Hours */}
                          <div className="px-5 py-3 bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-xl shadow-sm flex items-center gap-3">
                          <div className="w-12 h-12 rounded-full bg-purple-200 flex items-center justify-center">
                            <span className="text-purple-700 text-lg">🕒</span>
                          </div>
                          <div>
                            <div className="text-[11px] text-slate-600 uppercase tracking-wide">
                              Office Hours
                            </div>
                            <div className="text-2xl font-extrabold text-purple-800 leading-none">{totalHours}</div>
                            <div className="text-[10px] text-purple-600 mt-0.5">
                              {/* Worked in selected {reportType} */}
                            </div>
                          </div>
                        </div>

                                                  
                          


                        {/* Days count (not daily report) */}
                        {reportType !== "daily" && (
                          <div className="px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl text-sm shadow-sm flex items-center gap-3">
                            <div className="w-10 h-10 bg-amber-200 rounded-xl flex items-center justify-center">
                              <span className="text-amber-700 text-lg">📅</span>
                            </div>
                            <div>
                              <div className="text-[11px] text-slate-500">Days</div>
                              <div className="font-bold text-slate-800">{totalDays}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>

                  {/* DAILY REPORT VIEW */}
                  {/* DAILY REPORT VIEW */}
                  {/* DAILY REPORT VIEW */}
                  {reportType === "daily" ? (
                    <>
                      <motion.div
                        variants={itemVariants}
                        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
                      >
                        <div className="flex items-start justify-between gap-8">
                          {/* LEFT: Hours & Progress */}
                          <div className="flex-1 min-w-0">
                            {/* Get data for selected date */}
                            {(() => {
                              const item = analyticsData.daily_hours?.find((x) => x.date === selectedDate) || {};
                              const officeHoursStr= item.working_hours || analyticsData.total_hours || "0h 0m"; // First entry to last exit
                              const totalHoursStr =
                              totalHoursResult?.total_hours ||
                              totalHoursResult?.working_hours ||
                              "0h 0m";
                              const colorResult = getColorByHours(totalHoursStr, officeHoursStr);
                              
                              return (
                                <>
                                  <p className="text-xs text-slate-500">Office Hours</p>
                                  <p className="text-4xl font-extrabold text-slate-900">
                                    {officeHoursStr}
                                  </p>
                                  
                                  {/* Progress bar with NEW dual logic */}
                                  <div className="mt-4">
                                    <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                                      <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${calcProgress(totalHoursStr, officeHoursStr)}%` }}
                                        transition={{ duration: 0.9, ease: "easeOut" }}
                                        className="h-full rounded-full transition-all duration-300"
                                        style={{ backgroundColor: colorResult.stroke }}
                                      />
                                    </div>
                                  </div>
                                </>
                              );
                            })()}
                          </div>

                          {/* RIGHT: Metrics Table */}
                          <div className="flex-1 max-w-xs space-y-3 text-sm">
                            {(() => {
                              const item = analyticsData.daily_hours?.find((x) => x.date === selectedDate) || {};
                              const officeHoursStr= item.working_hours || analyticsData.total_hours || "0h 0m";
                              console.log("item.office_hours:", item?.working_hours);
                              const totalHoursStr =
                              totalHoursResult?.total_hours ||
                              totalHoursResult?.working_hours ||
                              "0h 0m";

                              console.log("Total Hours (API):", totalHoursStr);
                              const colorResult = getColorByHours(totalHoursStr, officeHoursStr);
                              const progress = calcProgress(totalHoursStr, officeHoursStr);


                              return (
                                <>
                                

                                  {/* First entry */}
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 uppercase">First entry</span>
                                    <span className="text-sm font-medium text-slate-700">
                                      {item.first_entry || "—"}
                                    </span>
                                  </div>

                                  {/* Last exit */}
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 uppercase">Last exit</span>
                                    <span className="text-sm font-medium text-slate-700">
                                      {item.last_event || "—"}
                                    </span>
                                  </div>

                                  {/* Circle progress with NEW color logic */}
                                  <div className="flex items-center justify-between pt-2">
                                    <span className="text-xs text-slate-500 uppercase">Progress</span>
                                    <CircleProgress 
                                      progress={progress} 
                                      strokeColor={colorResult.stroke}
                                    />
                                  </div>
                                </>
                              );
                            })()}
                          </div>
                        </div>
                      </motion.div>

                      {/* ENTRY/EXIT TIMELINE WITH PHOTO PREVIEW */}
                      
                      {/* ENTRY/EXIT TIMELINE */}
                      <motion.div
                        variants={itemVariants}
                        className="relative rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
                      >
                        <h4 className="text-lg font-bold text-slate-900 mb-4">Entry & Exit Timeline</h4>

                        {loadingEntries ? (
                          <div className="py-12 flex flex-col items-center gap-3 text-slate-500">
                            <FiLoader className="animate-spin text-blue-500 text-3xl" />
                            <p className="text-sm">Loading timeline…</p>
                          </div>
                        ) : !entriesData || entriesData.total === 0 ? (
                          <div className="py-12 text-center text-slate-500">
                            <FiAlertCircle className="text-3xl text-slate-300 mx-auto mb-3" />
                            <p>No entry/exit records for this date.</p>
                          </div>
                        ) : (
                          <div className="space-y-3">
                            {entriesData.records.map((record, idx) => {
                              const isEntry = record.camera.toLowerCase() === 'entry';
                              return (
                                <motion.div
                                  key={idx}
                                  initial={{ opacity: 0, x: -20 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: idx * 0.05 }}
                                  className={`flex items-center gap-4 p-4 rounded-xl border-2 ${
                                    isEntry
                                      ? 'bg-emerald-50 border-emerald-200'
                                      : 'bg-red-50 border-red-200'
                                  }`}
                                >
                                  {/* Photo Thumbnail */}
                                  {record.photo && (
                                    <div
                                      className="w-16 h-16 rounded-lg overflow-hidden border-2 border-white shadow-md cursor-pointer hover:scale-105 hover:shadow-lg transition-all duration-200"
                                      onClick={() => setPhotoPreview({ url: record.photo })}
                                      title="Click to preview image"
                                    >
                                      <img
                                        src={record.photo}
                                        alt={`${record.camera} photo`}
                                        className="w-full h-full object-cover hover:brightness-110 transition-all duration-200"
                                      />
                                    </div>
                                  )}

                                  {/* Details */}
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                                      <span
                                        className={`px-3 py-1 rounded-full text-xs font-bold ${
                                          isEntry
                                            ? 'bg-emerald-200 text-emerald-800'
                                            : 'bg-red-200 text-red-800'
                                        }`}
                                      >
                                        {record.camera}
                                      </span>
                                      <span className="text-sm text-slate-600 truncate" title={record.name}>
                                        {record.name}
                                      </span>
                                    </div>
                                    <div className="text-xs text-slate-500">
                                      {record.date} • {record.time}
                                    </div>
                                  </div>

                                  {/* Time badge */}
                                  <div
                                    className={`px-4 py-2 rounded-lg font-mono font-bold text-lg ${
                                      isEntry
                                        ? 'bg-emerald-100 text-emerald-800'
                                        : 'bg-red-100 text-red-800'
                                    }`}
                                  >
                                    {record.time}
                                  </div>
                                </motion.div>
                              );
                            })}
                          </div>
                        )}
                      </motion.div>
                    </>
                  ) : (
                    // WEEKLY or MONTHLY SUMMARY VIEW - SAME LOGIC AS DAILY (not structure)
                    <motion.div variants={itemVariants} className="space-y-4">
                      <div className="grid md:grid-cols-2 gap-4">
                        {/* LineChart: Hours trend */}
                        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                          <div className="flex items-center justify-between mb-3">
                            <div>
                              <p className="text-xs text-slate-500 uppercase tracking-wide">Trend</p>
                              <p className="text-sm font-semibold text-slate-900">Hours time</p>
                            </div>
                            <div className="text-xs text-slate-500">Animated</div>
                          </div>
                          <div style={{ width: "100%", height: 220 }}>
                            <ResponsiveContainer>
                              <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                                <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
                                <Tooltip formatter={(value) => `${value} hrs`} />
                                <Line
                                  type="monotone"
                                  dataKey="hoursNum"
                                  stroke="#6366f1"
                                  strokeWidth={3}
                                  dot={{ r: 3 }}
                                  activeDot={{ r: 6 }}
                                  animationDuration={800}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        {/* BarChart: Daily vs target */}

                        {/* BarChart: Daily vs target */}
                        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                          <div className="flex items-center justify-between mb-3">
                            <div>
                              <p className="text-xs text-slate-500 uppercase tracking-wide">
                                Comparison
                              </p>
                              <p className="text-sm font-semibold text-slate-900">
                                Daily vs target
                              </p>
                            </div>
                          </div>

                          <div style={{ width: "100%", height: 220 }}>
                            <ResponsiveContainer>
                              <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                                <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
                                <Tooltip formatter={(value) => `${value} hrs`} />

                                <Bar dataKey="hoursNum" barSize={18} animationDuration={700}>
                                  {chartData.map((entry, idx) => (
                                    <Cell
                                      key={`cell-${idx}`}
                                      fill={entry.color} // ✅ SAME COLOR AS DAILY DETAILS
                                    />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                      </div>

                      
                      {/* DAILY DETAILS - WITH ENTRY/EXIT PHOTOS */}
                      {/* DAILY DETAILS - WITH ENTRY/EXIT PHOTOS */}
                      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <p className="text-xs text-slate-500 uppercase tracking-wide">Daily Details</p>
                            <p className="text-sm font-semibold text-slate-900">Per-day breakdown</p>
                          </div>
                          <div className="text-xs text-slate-500"></div>
                        </div>

                        <div className="grid md:grid-cols-3 gap-3">
                          {dailyHours.map((item, i) => {
                            const date = item.date || "N/A";
                            const totalHoursStr = item.working_hours || "0h 0m";

                            const matchedDay =
                              analyticsData.daily_hours?.find((x) => x.date === date) || {};
                            const officeHoursStr =
                              matchedDay.office_hours || matchedDay.working_hours || "8h 0m";

                            const colorResult = getColorByHours(totalHoursStr, officeHoursStr);
                            const progress = calcProgress(totalHoursStr, officeHoursStr);

                            // Get first entry and last exit photos for this date
                            const dayPhotos = employeePhotos[date] || { entries: [], exits: [] };
                            const firstEntryPhoto = dayPhotos.entries.length > 0 
                              ? dayPhotos.entries[0].photo 
                              : null;
                            const lastExitPhoto = dayPhotos.exits.length > 0 
                              ? dayPhotos.exits[dayPhotos.exits.length - 1].photo 
                              : null;

                            return (
                              <div key={i} className="p-3 rounded-xl border border-slate-100 hover:shadow-md transition">
                                <div className="flex items-center justify-between mb-2">
                                  <div>
                                    <p className="text-sm font-semibold text-slate-900">{item.date}</p>
                                    <p className="text-xs font-bold text-slate-900">{officeHoursStr}</p>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-xs text-slate-500">Progress</div>
                                    <div className="font-bold text-slate-900">{Math.round(progress)}%</div>
                                  </div>
                                </div>

                                {/* Entry/Exit with Eye Icons - Parallel Layout */}
                                <div className="grid grid-cols-2 gap-2 mb-2">
                                  {/* Entry */}
                                  <div className="flex items-center gap-1 p-2 rounded-lg bg-emerald-50 border border-emerald-200">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-[10px] text-emerald-600 font-semibold uppercase">Entry</p>
                                      <p className="text-xs font-bold text-emerald-800 truncate">
                                        {item.first_entry || "—"}
                                      </p>
                                    </div>
                                    {firstEntryPhoto ? (
                                      <button
                                        onClick={() => setPhotoPreview({ url: firstEntryPhoto })}
                                        className="flex-shrink-0 w-7 h-7 rounded-full bg-emerald-600 hover:bg-emerald-700 flex items-center justify-center transition-all hover:scale-110"
                                        title="View entry photo"
                                      >
                                        <FiImage className="text-white text-sm" />
                                      </button>
                                    ) : (
                                      <div
                                        className="flex-shrink-0 w-7 h-7 rounded-full bg-slate-300 flex items-center justify-center"
                                        title="No photo available"
                                      >
                                        <FiX className="text-slate-500 text-xs" />
                                      </div>
                                    )}
                                  </div>

                                  {/* Exit */}
                                  <div className="flex items-center gap-1 p-2 rounded-lg bg-red-50 border border-red-200">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-[10px] text-red-600 font-semibold uppercase">Exit</p>
                                      <p className="text-xs font-bold text-red-800 truncate">
                                        {item.last_event || "—"}
                                      </p>
                                    </div>
                                    {lastExitPhoto ? (
                                      <button
                                        onClick={() => setPhotoPreview({ url: lastExitPhoto })}
                                        className="flex-shrink-0 w-7 h-7 rounded-full bg-red-600 hover:bg-red-700 flex items-center justify-center transition-all hover:scale-110"
                                        title="View exit photo"
                                      >
                                        <FiImage className="text-white text-sm" />
                                      </button>
                                    ) : (
                                      <div
                                        className="flex-shrink-0 w-7 h-7 rounded-full bg-slate-300 flex items-center justify-center"
                                        title="No photo available"
                                      >
                                        <FiX className="text-slate-500 text-xs" />
                                      </div>
                                    )}
                                  </div>
                                </div>

                                {/* Progress Bar */}
                                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${progress}%` }}
                                    transition={{ duration: 0.8, ease: "easeOut" }}
                                    style={{
                                      height: "100%",
                                      background: colorResult.stroke,
                                    }}
                                  />
                                </div>

                                {/* Status Label */}
                                <div className="mt-2 text-xs">
                                  <span className="inline-flex items-center gap-2 px-2 py-1 rounded-full bg-white border">
                                    <span
                                      style={{
                                        width: 8,
                                        height: 8,
                                        background: colorResult.stroke,
                                        display: "inline-block",
                                        borderRadius: 4,
                                      }}
                                    />
                                    <span className="font-medium">{colorResult.label}</span>
                                  </span>
                                </div>


                              </div>
                            );
                          }) || (
                            <div className="col-span-full py-12 text-center text-slate-500">
                              <FiAlertCircle className="text-3xl text-slate-300 mx-auto mb-3" />
                              <p>No daily data available</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>


                  )}
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* FULLSCREEN PHOTO MODAL */}
          
          <AnimatePresence>
            {photoPreview.url && (
              <motion.div
                ref={modalRef}
                className="fixed inset-0 z-[99999] bg-black/80 flex items-center justify-center p-4"
                style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0 }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={(e) => {
                  e.stopPropagation();
                  setPhotoPreview({ url: null });
                }}
              >
                <motion.div
                  className="relative max-w-[90vw] max-h-[90vh] rounded-2xl shadow-2xl overflow-hidden bg-white"
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.5, opacity: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <img
                    src={photoPreview.url}
                    alt="preview"
                    className="object-contain max-w-full max-h-[85vh] rounded-xl"
                  />

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setPhotoPreview({ url: null });
                    }}
                    className="absolute top-3 right-3 bg-black/70 text-white p-3 rounded-full hover:bg-black/90 transition-all hover:scale-110"
                    aria-label="Close preview"
                  >
                    <FiX size={20} />
                  </button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>



        </motion.div>
      )}
    </AnimatePresence>
  );
};






// ============================================
// PART 3 OF 3: MAIN COMPONENT
// Copy this entire section after Part 2
// ============================================

// Main Component - UPDATED
export default function Report() {
  const { employeeId } = useParams();

  // State - UPDATED
  const [employee, setEmployee] = useState(null);
  const [logs, setLogs] = useState([]);
  const [totalLogs, setTotalLogs] = useState(0);
  const [currentLogBatch, setCurrentLogBatch] = useState(1);
  const [workingHoursData, setWorkingHoursData] = useState([]);
  const [totalWorkingHours, setTotalWorkingHours] = useState(0);
  const [pageHours, setPageHours] = useState(1);
  const [filterType, setFilterType] = useState("both");
  const [selectedDate, setSelectedDate] = useState("");  // CHANGED
  const [reportType, setReportType] = useState("weekly");  // CHANGED
  const [expandedImg, setExpandedImg] = useState(null);
  const [expandedDates, setExpandedDates] = useState({});
  const [loading, setLoading] = useState(true);
  const [logsLoading, setLogsLoading] = useState(false);
  const [hoursLoading, setHoursLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAnalytics, setShowAnalytics] = useState(false);  // NEW

  // Fetch employee
  useEffect(() => {
    let mounted = true;
    const fetchEmployee = async () => {
      setLoading(true);
      setError("");
      try {
        const resp = await axios.get(`${BASE_URL}/api/employees`);
        const employees = resp.data.employees || resp.data || [];
        const found = employees.find(
          (e) => String(e.emp_id || e.employee_id) === String(employeeId)
        );
        if (mounted) setEmployee(found || null);
      } catch (err) {
        console.error("fetchEmployee:", err);
        if (mounted) setError("Failed to fetch employee data.");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchEmployee();
    return () => { mounted = false; };
  }, [employeeId]);

  // Fetch working hours - UPDATED
  const fetchWorkingHours = useCallback(async (page = 1) => {
    if (!employeeId) return;
    setHoursLoading(true);
    setError("");
    try {
      const offset = (page - 1) * ITEMS_PER_PAGE;
      const params = { 
        emp_id: employeeId, 
        limit: ITEMS_PER_PAGE, 
        offset,
        report_type: reportType
      };

      // Only add date filter for daily report type in the main table
      // NOT when viewing weekly/monthly summaries
      if (selectedDate && reportType === "daily") {
        params.from_date = selectedDate;
        params.to_date = selectedDate;
      }

      const resp = await axios.get(`${BASE_URL}/api/calculate-working-hours-full`, { params });
      console.log('Fetched data:', resp.data); // Debug
      setWorkingHoursData(resp.data.working_hours || []);
      setTotalWorkingHours(resp.data.total || 0);
    } catch (err) {
      console.error("fetchWorkingHours:", err);
      setError("Failed to fetch working hours.");
      setWorkingHoursData([]);
      setTotalWorkingHours(0);
    } finally {
      setHoursLoading(false);
    }
  }, [employeeId, reportType, selectedDate]);

  // Fetch logs - UPDATED
  const fetchLogs = useCallback(async (batchNumber = 1) => {
    if (!employeeId) return;
    setLogsLoading(true);
    setError("");
    try {
      const offset = (batchNumber - 1) * LOGS_PER_BATCH;
      const params = { 
        emp_id: employeeId, 
        limit: LOGS_PER_BATCH, 
        offset,
        camera: filterType !== "both" ? filterType : undefined
      };

      // Add date filter
      if (selectedDate) {
        params.from_date = selectedDate;
        params.to_date = selectedDate;
      }
      
      const resp = await axios.get(`${BASE_URL}/api/logs`, { params });
      
      if (batchNumber === 1) {
        setLogs(resp.data.logs || []);
      } else {
        setLogs(prev => [...prev, ...(resp.data.logs || [])]);
      }
      
      setTotalLogs(resp.data.total || 0);
      setCurrentLogBatch(batchNumber);
    } catch (err) {
      console.error("fetchLogs:", err);
      setError("Failed to fetch logs.");
    } finally {
      setLogsLoading(false);
    }
  }, [employeeId, selectedDate, filterType]);

  // Effects
  useEffect(() => {
    if (!employee) return;
    fetchWorkingHours(pageHours);
  }, [employee, pageHours, fetchWorkingHours]);

  useEffect(() => {
    if (!employee) return;
    fetchLogs(1);
  }, [employee, fetchLogs]);

  useEffect(() => {
    setPageHours(1);
    if (employee) {
      fetchWorkingHours(1);
      fetchLogs(1);
    }
  }, [filterType, reportType, selectedDate, employee, fetchWorkingHours, fetchLogs]);  // UPDATED

  // Computed values
  const filteredLogs = useMemo(() => {
    let arr = (logs || []).slice();
    if (filterType === "Entry") {
      arr = arr.filter((l) => (l.camera || "").toLowerCase() === "entry");
    }
    if (filterType === "Exit") {
      arr = arr.filter((l) => (l.camera || "").toLowerCase() === "exit");
    }
    return arr;
  }, [logs, filterType]);

  const logsByDate = useMemo(() => {
    return filteredLogs.reduce((acc, log) => {
      const d = safeDateString(log.date || log.timestamp);
      if (!d) return acc;
      if (!acc[d]) acc[d] = [];
      acc[d].push(log);
      return acc;
    }, {});
  }, [filteredLogs]);

  const allDates = useMemo(() => {
    return Object.keys(logsByDate).sort((a, b) => b.localeCompare(a));
  }, [logsByDate]);

  const summarizedHours = useMemo(() => {
    if (reportType === "daily") {
      return (workingHoursData || []).map((r) => ({
        date: r.date,
        label: r.date,
        working_hours: r.working_hours,
        status: r.status,  // Add this for Holiday/Absent detection
        isAbsent: r.status === "Absent",
        weekend: isWeekend(r.date),
        holidayReason: HOLIDAYS[r.date] || null,
        hasLogs: !!(logsByDate[r.date] && logsByDate[r.date].length)
      }));
    }
    return (workingHoursData || []).map((r) => ({
      ...r,
      label: r.week || r.month || r.date,
      working_hours: r.total_hours || r.working_hours,
      status: r.status,  // Add this
      isAbsent: false,
      weekend: false,
      holidayReason: null,
      hasLogs: true
    }));
  }, [workingHoursData, reportType, logsByDate]);
  // Handlers - UPDATED
  const handleReset = () => {
    setSelectedDate("");
    setFilterType("both");
    setPageHours(1);
    setReportType("weekly");
  };

  const toggleDateExpand = (date) => {
    setExpandedDates((p) => ({ ...p, [date]: !p[date] }));
  };

  const handleViewAnalytics = (type) => {  // NEW
    setReportType(type);
    setShowAnalytics(true);
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay message={error} />;

  return (
    <motion.div
      className="min-h-screen bg-gray-50 p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="max-w-7xl mx-auto">
        <EmployeeHeader employee={employee} />
        
        <FilterControls
          reportType={reportType}
          setReportType={setReportType}
          selectedDate={selectedDate}
          setSelectedDate={setSelectedDate}
          filterType={filterType}
          setFilterType={setFilterType}
          onViewAnalytics={handleViewAnalytics}
        />

        <WorkingHoursTable
          data={summarizedHours}
          loading={hoursLoading}
          reportType={reportType}
          totalHours={totalWorkingHours}
          currentPage={pageHours}
          onPageChange={setPageHours}
        />

        <motion.div
          className="bg-white rounded-xl shadow-md border border-gray-200 p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <FiActivity className="text-blue-600 text-xl" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-800">Attendance Logs</h3>
                <p className="text-xs text-gray-500">Daily entry and exit records</p>
              </div>
            </div>
            {logsLoading && (
              <div className="flex items-center gap-2">
                <FiLoader className="animate-spin text-blue-600" />
                <span className="text-sm text-gray-500">Loading...</span>
              </div>
            )}
          </div>

          {allDates.length === 0 ? (
            <div className="text-center py-16">
              <FiCalendar className="text-5xl text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg font-medium">No logs found</p>
              <p className="text-gray-400 text-sm mt-2">Try adjusting your filters</p>
            </div>
          ) : (
            <>
              {allDates.map((date) => (
                <DayLogsCard
                  key={date}
                  date={date}
                  logs={logsByDate[date]}
                  employee={employee}
                  expandedDates={expandedDates}
                  onToggleExpand={toggleDateExpand}
                  onImageClick={setExpandedImg}
                />
              ))}

              {logs.length < totalLogs && (
                <div className="flex justify-center mt-6">
                  <button
                    className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    onClick={() => fetchLogs(currentLogBatch + 1)}
                    disabled={logsLoading}
                  >
                    {logsLoading ? (
                      <>
                        <FiLoader className="animate-spin" />
                        <span>Loading...</span>
                      </>
                    ) : (
                      <>
                        <span>Load More Logs</span>
                        <span className="px-2 py-0.5 bg-blue-500 rounded text-xs">
                          {totalLogs - logs.length} more
                        </span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </motion.div>
      </div>

      <AnimatePresence>
        {expandedImg && (
          <ImageModal imageUrl={expandedImg} onClose={() => setExpandedImg(null)} />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showAnalytics && (
          <AnalyticsModal
            key={`analytics-${reportType}-${selectedDate}`}
            isOpen={showAnalytics}
            onClose={() => {
              setShowAnalytics(false);
              // Clear date filter if it was daily view
              if (reportType === "daily") {
                setSelectedDate("");
              }
              // Force refresh
              setTimeout(() => {
                fetchWorkingHours(1);
                setPageHours(1);
              }, 100);
            }}
            reportType={reportType}
            employeeId={employeeId}
            employee={employee}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}