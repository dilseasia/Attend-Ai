import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Calendar, Camera, Download, RefreshCw, Users, Eye, X, Clock } from 'lucide-react';

export default function Attendance() {
  const [logs, setLogs] = useState([]);
  const [employees, setEmployees] = useState([]);
  const [searchName, setSearchName] = useState('');
  const [searchDate, setSearchDate] = useState('');
  const [cameraType, setCameraType] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedLog, setSelectedLog] = useState(null);
  const [employeePhoto, setEmployeePhoto] = useState(null);
  const [loadingPhoto, setLoadingPhoto] = useState(false);

  useEffect(() => {
    setLoading(true);
    
    // Fetch logs
    fetch('http://10.8.11.183:8000/api/logs')
      .then(res => res.json())
      .then(data => {
        setLogs(data.logs || []);
      })
      .catch(err => console.error('Failed to fetch logs:', err));
    
    // Fetch employees for photos
    fetch('http://10.8.11.183:8000/api/employees')
      .then(res => res.json())
      .then(data => {
        setEmployees(data.employees || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch employees:', err);
        setLoading(false);
      });
  }, []);

  const filteredLogs = logs.filter((log) => {
    const matchesName = log.name.toLowerCase().includes(searchName.toLowerCase());
    const matchesDate = searchDate ? log.date === searchDate : true;
    const matchesCamera = cameraType ? log.camera.toLowerCase() === cameraType.toLowerCase() : true;
    return matchesName && matchesDate && matchesCamera;
  });

  const handleExport = () => {
    const headers = ['Name', 'Employee ID', 'Date', 'Time', 'Camera'];
    const csvRows = [
      headers.join(','),
      ...filteredLogs.map(log =>
        [log.name, log.emp_id, log.date, log.time, log.camera].join(',')
      )
    ];
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const resetFilters = () => {
    setSearchName('');
    setSearchDate('');
    setCameraType('');
  };

  // Get employee photo by emp_id
  const getEmployeePhoto = (empId) => {
    const employee = employees.find(emp => emp.emp_id === empId);
    return employee ? employee.image_url : null;
  };

  // 📸 Fetch employee photo from API matching exact timestamp
  const fetchEmployeePhotoFromAPI = async (empId, date, time, camera) => {
    setLoadingPhoto(true);
    setEmployeePhoto(null);
    
    try {
      const res = await fetch(
        `http://10.8.11.183:8000/api/employee-entries-with-photos?emp_id=${empId}&type=all&date=${date}`
      );
      const data = await res.json();
      
      // Find the exact photo matching the time and camera
      if (data?.records && data.records.length > 0) {
        // Find exact match by time and camera
        const matchedPhoto = data.records.find(record => {
          const recordTime = record.time || '';
          const recordCamera = (record.camera || '').toLowerCase();
          const logCamera = (camera || '').toLowerCase();
          
          // Match both time and camera type
          return recordTime === time && recordCamera === logCamera;
        });
        
        if (matchedPhoto) {
          setEmployeePhoto(matchedPhoto);
        } else {
          // If exact match not found, try to find closest time match
          console.log('Exact match not found, trying closest match...');
          const closestMatch = data.records.find(record => {
            const recordTime = record.time || '';
            const recordCamera = (record.camera || '').toLowerCase();
            const logCamera = (camera || '').toLowerCase();
            
            // Match camera and check if time is close (within same minute)
            const recordMinutes = recordTime.substring(0, 5); // HH:MM
            const logMinutes = time.substring(0, 5); // HH:MM
            
            return recordCamera === logCamera && recordMinutes === logMinutes;
          });
          
          setEmployeePhoto(closestMatch || null);
        }
      } else {
        setEmployeePhoto(null);
      }
    } catch (err) {
      console.error('Error fetching employee photo:', err);
      setEmployeePhoto(null);
    } finally {
      setLoadingPhoto(false);
    }
  };

  // Handle View button click
  const handleViewClick = (log) => {
    setSelectedLog(log);
    fetchEmployeePhotoFromAPI(log.emp_id, log.date, log.time, log.camera);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
            Attendance Dashboard
          </h1>
          <p className="text-gray-600">Monitor and manage employee attendance records</p>
        </motion.div>

        {/* Filters Section */}
        <motion.div
          className="grid md:grid-cols-4 sm:grid-cols-2 gap-4 mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          {/* Search */}
          <input
            type="text"
            placeholder="Search by name"
            className="border px-4 py-3 rounded-full bg-white text-gray-900 shadow focus:ring-2 focus:ring-blue-200 transition"
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
          />

          {/* Date */}
          <input
            type="date"
            className="border px-4 py-3 rounded-full bg-white text-gray-900 shadow focus:ring-2 focus:ring-blue-200 transition"
            value={searchDate}
            onChange={(e) => setSearchDate(e.target.value)}
          />

          {/* Camera */}
          <select
            className="border px-4 py-3 rounded-full bg-white text-gray-900 shadow focus:ring-2 focus:ring-blue-200 transition"
            value={cameraType}
            onChange={(e) => setCameraType(e.target.value)}
          >
            <option value="">All Cameras</option>
            <option value="entry">Entry</option>
            <option value="exit">Exit</option>
          </select>

          {/* Export Button */}
          <button
            onClick={handleExport}
            className="bg-green-600 text-white px-4 py-3 rounded-full shadow hover:bg-green-700 transition font-semibold"
          >
            ⬇ Export CSV
          </button>
        </motion.div>

        {/* Table Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100"
        >
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="relative">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <div className="absolute inset-0 animate-ping rounded-full h-12 w-12 border-b-2 border-blue-400 opacity-20"></div>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gradient-to-r from-[#5A8BFF] via-[#7A6CFF] to-[#9A5BFF]">
                  <tr>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Name</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Employee ID</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Date</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Time</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Camera</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-white">Action</th>
                  </tr>
                </thead>

                <tbody>
                  <AnimatePresence>
                    {filteredLogs.length === 0 ? (
                      <tr>
                        <td colSpan="6" className="text-center py-12">
                          <div className="flex flex-col items-center justify-center text-gray-400">
                            <Users size={48} className="mb-4 opacity-50" />
                            <p className="text-lg font-medium">No records found</p>
                            <p className="text-sm">Try adjusting your filters</p>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      filteredLogs.map((log, idx) => {
                        const photoUrl = getEmployeePhoto(log.emp_id);
                        
                        return (
                          <motion.tr
                            key={`${log.emp_id}_${idx}`}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            transition={{ duration: 0.3, delay: Math.min(idx * 0.02, 0.3) }}
                            className="border-t border-gray-100 hover:bg-blue-50/50 transition-colors"
                          >
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                {photoUrl ? (
                                  <img 
                                    src={photoUrl} 
                                    alt={log.name}
                                    className="w-10 h-10 rounded-full object-cover shadow-md border-2 border-white"
                                    onError={(e) => {
                                      e.target.style.display = 'none';
                                      e.target.nextElementSibling.style.display = 'flex';
                                    }}
                                  />
                                ) : null}
                                <div 
                                  className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center text-white font-semibold shadow-md"
                                  style={{ display: photoUrl ? 'none' : 'flex' }}
                                >
                                  {log.name.charAt(0).toUpperCase()}
                                </div>
                                <span className="font-medium text-gray-900">{log.name}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-gray-600 font-mono text-sm">{log.emp_id}</td>
                            <td className="px-6 py-4 text-gray-600">{log.date}</td>
                            <td className="px-6 py-4 text-gray-600 font-mono text-sm">{log.time}</td>
                            <td className="px-6 py-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1 ${
                                log.camera.toLowerCase() === 'entry' 
                                  ? 'bg-green-100 text-green-700' 
                                  : 'bg-orange-100 text-orange-700'
                              }`}>
                                <div className={`w-2 h-2 rounded-full ${
                                  log.camera.toLowerCase() === 'entry' ? 'bg-green-500' : 'bg-orange-500'
                                }`}></div>
                                {log.camera.charAt(0).toUpperCase() + log.camera.slice(1)}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <button
                                onClick={() => handleViewClick(log)}
                                className="text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1 transition-colors hover:gap-2"
                              >
                                <Eye size={16} />
                                View
                              </button>
                            </td>
                          </motion.tr>
                        );
                      })
                    )}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          )}
          
          {/* Pagination Info */}
          {!loading && filteredLogs.length > 0 && (
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 flex items-center justify-between">
              <p className="text-sm text-gray-600">
                Showing <span className="font-semibold text-gray-900">{filteredLogs.length}</span> records
              </p>
            </div>
          )}
        </motion.div>

        {/* Modal for viewing details with API photo */}
        <AnimatePresence>
          {selectedLog && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              onClick={() => {
                setSelectedLog(null);
                setEmployeePhoto(null);
              }}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                transition={{ type: "spring", duration: 0.5 }}
                className="bg-white rounded-2xl p-8 max-w-lg w-full shadow-2xl relative"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => {
                    setSelectedLog(null);
                    setEmployeePhoto(null);
                  }}
                  className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 transition-colors"
                >
                  <X size={20} className="text-gray-500" />
                </button>
                
                {/* Employee Header */}
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900">{selectedLog.name}</h3>
                  <p className="text-gray-500 font-mono text-sm">{selectedLog.emp_id}</p>
                </div>

                {/* Photo Section */}
                <div className="mb-6">
                  {loadingPhoto ? (
                    <div className="flex items-center justify-center h-64 bg-gray-100 rounded-xl">
                      <div className="relative">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                        <div className="absolute inset-0 animate-ping rounded-full h-12 w-12 border-b-2 border-blue-400 opacity-20"></div>
                      </div>
                    </div>
                  ) : employeePhoto?.photo ? (
                    <div className="rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={employeePhoto.photo}
                        alt={selectedLog.name}
                        className="w-full h-auto max-h-96 object-cover"
                        onError={(e) => {
                          e.target.src = "https://via.placeholder.com/400x300?text=Photo+Not+Available";
                        }}
                      />
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-gray-100 to-gray-200 rounded-xl">
                      <Camera size={48} className="text-gray-400 mb-3" />
                      <p className="text-gray-500 font-medium">No photo available</p>
                      <p className="text-gray-400 text-sm">for this timestamp</p>
                    </div>
                  )}
                </div>
                
                {/* Details Section */}
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 font-medium flex items-center gap-2">
                      <Calendar size={16} />
                      Date
                    </span>
                    <span className="text-gray-900 font-semibold">{selectedLog.date}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 font-medium flex items-center gap-2">
                      <Clock size={16} />
                      Time
                    </span>
                    <span className="text-gray-900 font-semibold font-mono">{selectedLog.time}</span>
                  </div>
                  <div className="flex justify-between items-center py-3">
                    <span className="text-gray-500 font-medium flex items-center gap-2">
                      <Camera size={16} />
                      Camera
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1 ${
                      selectedLog.camera.toLowerCase() === 'entry' 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-orange-100 text-orange-700'
                    }`}>
                      <div className={`w-2 h-2 rounded-full ${
                        selectedLog.camera.toLowerCase() === 'entry' ? 'bg-green-500' : 'bg-orange-500'
                      }`}></div>
                      {selectedLog.camera.charAt(0).toUpperCase() + selectedLog.camera.slice(1)}
                    </span>
                  </div>
                </div>
                
                <button
                  onClick={() => {
                    setSelectedLog(null);
                    setEmployeePhoto(null);
                  }}
                  className="mt-6 w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl"
                >
                  Close
                </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}