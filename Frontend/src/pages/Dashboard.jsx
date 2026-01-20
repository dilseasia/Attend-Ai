import React, { useEffect, useState } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { Camera, Users, Calendar, TrendingUp, Clock, RotateCcw, CalendarDays, ChevronLeft, ChevronRight } from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

export default function Dashboard() {
  const [stats, setStats] = useState({
    totalPresent: 0,
    entryCount: 0,
    exitCount: 0,
    knownEmployees: 0,
  });
  const [monthlyChartData, setMonthlyChartData] = useState([]);
  const [currentMonthName, setCurrentMonthName] = useState('');
  const [workingHoursData, setWorkingHoursData] = useState([]);
  const [lastUpdated, setLastUpdated] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const controls = useAnimation();
  
  // Separate date navigation states
  const [selectedDailyDate, setSelectedDailyDate] = useState(new Date());
  const [selectedMonthDate, setSelectedMonthDate] = useState(new Date());
  const [dailyViewMode, setDailyViewMode] = useState('today'); // 'today' or 'custom'

  const BASE_URL = "http://10.8.11.183:8000";

  // Fetch stats whenever dates change
  useEffect(() => {
    fetchStats();
  }, [selectedDailyDate, selectedMonthDate, dailyViewMode]);

  // Auto-refresh only when in 'today' mode
  useEffect(() => {
    if (dailyViewMode === 'today') {
      const interval = setInterval(() => {
        fetchStats();
      }, 30000);
      return () => clearInterval(interval);
    }
  }, [dailyViewMode]);

  const fetchStats = async () => {
    setIsLoading(true);
    const targetDate = dailyViewMode === 'today' ? new Date() : selectedDailyDate;
    
    // Format date properly to avoid timezone issues
    const year = targetDate.getFullYear();
    const month = String(targetDate.getMonth() + 1).padStart(2, '0');
    const day = String(targetDate.getDate()).padStart(2, '0');
    const todayStr = `${year}-${month}-${day}`;
    
    const monthYear = selectedMonthDate.getFullYear();
    const monthIdx = selectedMonthDate.getMonth();

    setCurrentMonthName(selectedMonthDate.toLocaleString('default', { month: 'long', year: 'numeric' }));

    try {
      const [logsRes, empRes] = await Promise.all([
        fetch(`${BASE_URL}/api/logs/all`).then(r => r.json()),
        fetch(`${BASE_URL}/api/employees`).then(r => r.json())
      ]);

      const allLogs = logsRes.logs || logsRes || [];
      const employees = empRes.employees || empRes || [];

      const logsToday = allLogs.filter((log) => log.date === todayStr);
      const entryCount = logsToday.filter((l) => l.camera === "Entry").length;
      const exitCount = logsToday.filter((l) => l.camera === "Exit").length;
      const knownEmployees = employees.length;
      const totalPresent = new Set(
        logsToday.map((l) => l.emp_id || l.employee_id).filter(id => id)
      ).size;

      setStats({ totalPresent, entryCount, exitCount, knownEmployees });

      processMonthlyData(allLogs, monthYear, monthIdx);
      processWorkingHours(allLogs, todayStr);

      setLastUpdated(new Date().toLocaleTimeString());
      controls.start({ scale: [1, 1.05, 1], transition: { duration: 0.8 } });
    } catch (err) {
      console.error("❌ Dashboard fetch error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const processMonthlyData = (allLogs, year, monthIdx) => {
    const dailyCounts = {};

    allLogs.forEach(log => {
      const parts = log.date.split('-');
      const logYear = parseInt(parts[0]);
      const logMonthIdx = parseInt(parts[1]) - 1;
      const logDay = parseInt(parts[2]);

      if (logYear === year && logMonthIdx === monthIdx) {
        if (!dailyCounts[logDay]) {
          dailyCounts[logDay] = new Set();
        }
        const empId = log.emp_id || log.employee_id;
        if(empId) dailyCounts[logDay].add(empId);
      }
    });

    const chartData = Object.keys(dailyCounts)
      .sort((a, b) => parseInt(a) - parseInt(b))
      .map(day => ({
        day: `${day}`,
        Present: dailyCounts[day].size,
      }));

    setMonthlyChartData(chartData);
  };

  const processWorkingHours = (allLogs, todayStr) => {
    const hourCounts = Array(24).fill(0).map((_, i) => ({ hour: `${i}:00`, count: 0 }));
    
    allLogs
      .filter(log => log.date === todayStr && log.time)
      .forEach(log => {
        const hour = parseInt(log.time.split(':')[0]);
        if (hour >= 0 && hour < 24) {
          hourCounts[hour].count++;
        }
      });

    setWorkingHoursData(hourCounts.filter(h => h.count > 0));
  };

  const handleResetToToday = () => {
    setSelectedDailyDate(new Date());
    setDailyViewMode('today');
  };

  const handleDailyDateChange = (e) => {
    const dateString = e.target.value;
    if (!dateString) return;
    
    const [year, month, day] = dateString.split('-').map(Number);
    
    const selectedDate = new Date(year, month - 1, day, 12, 0, 0);
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const selectedDateMidnight = new Date(year, month - 1, day, 0, 0, 0);
    
    if (selectedDateMidnight <= today) {
      setSelectedDailyDate(selectedDate);
      if (selectedDateMidnight.getTime() === today.getTime()) {
        setDailyViewMode('today');
      } else {
        setDailyViewMode('custom');
      }
    }
  };

  const handleResetToCurrentMonth = () => {
    setSelectedMonthDate(new Date());
  };

  const handleMonthChange = (e) => {
    const [year, month] = e.target.value.split('-');
    const newDate = new Date(parseInt(year), parseInt(month) - 1, 1);
    setSelectedMonthDate(newDate);
  };

  // Month navigation functions
  const navigateMonth = (direction) => {
    const newDate = new Date(selectedMonthDate);
    if (direction === 'prev') {
      newDate.setMonth(newDate.getMonth() - 1);
    } else {
      newDate.setMonth(newDate.getMonth() + 1);
    }
    setSelectedMonthDate(newDate);
  };

  const isToday = dailyViewMode === 'today';
  const isCurrentMonth = selectedMonthDate.getMonth() === new Date().getMonth() && 
                         selectedMonthDate.getFullYear() === new Date().getFullYear();

  const formatDateForInput = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  const formatMonthForInput = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    return `${year}-${month}`;
  };

  const getMaxDate = () => {
    const today = new Date();
    return formatDateForInput(today);
  };

  return (
    <motion.div
      className="min-h-screen p-8 text-gray-900 overflow-hidden relative"
      style={{
        background: "linear-gradient(135deg, #c7d2fe 0%, #f0f9ff 100%)",
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.div
        className="absolute inset-0 opacity-40 bg-gradient-to-br from-blue-200 via-purple-100 to-pink-100 blur-3xl"
        animate={{
          backgroundPosition: ["0% 0%", "100% 100%"],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />

      <motion.div
        className="relative z-10 flex items-center space-x-3 text-3xl font-extrabold text-blue-800 mb-8"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 80 }}
      >
        <motion.div
          initial={{ rotate: -10, scale: 0 }}
          animate={{ rotate: 0, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <Camera size={45} />
        </motion.div>
        <h1>AI Face Recognition Dashboard</h1>
      </motion.div>

      <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
        <AnimatedStatCard color="green" label="Present Today" value={stats.totalPresent} controls={controls} />
        <AnimatedStatCard color="blue" label="Total Employees" value={stats.knownEmployees} controls={controls} />
        <AnimatedStatCard color="purple" label="Entry Logs" value={stats.entryCount} controls={controls} />
        <AnimatedStatCard color="red" label="Exit Logs" value={stats.exitCount} controls={controls} />
      </div>

      <div className="relative z-10 grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        {/* MONTHLY ATTENDANCE CHART */}
        <motion.div
          className="bg-white/70 backdrop-blur-md shadow-2xl rounded-3xl p-6 border border-white/30 relative"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          {isLoading && (
            <motion.div
              className="absolute inset-0 bg-white/50 backdrop-blur-sm rounded-3xl flex items-center justify-center z-10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            </motion.div>
          )}

          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-blue-800 flex items-center">
              <TrendingUp className="mr-2" size={20} /> Monthly Attendance
            </h2>
            
            {/* Monthly Navigation Controls - < December 2025 > format */}
            <div className="flex items-center gap-1">
              {/* Previous Month Button */}
              <motion.button
                onClick={() => navigateMonth('prev')}
                className="p-2 bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-blue-200 hover:bg-blue-50 hover:border-blue-400 hover:shadow-lg transition-all duration-200 flex items-center justify-center w-10 h-10"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <ChevronLeft size={16} className="text-blue-700" />
              </motion.button>

              {/* Month Display */}
              <div className="px-4 py-2.5 bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border-2 border-blue-300 text-lg font-bold text-blue-800 min-w-[160px] text-center">
                {currentMonthName}
              </div>

              {/* Next Month Button */}
              <motion.button
                onClick={() => navigateMonth('next')}
                className={`p-2 bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-blue-200 hover:bg-blue-50 hover:border-blue-400 hover:shadow-lg transition-all duration-200 flex items-center justify-center w-10 h-10 ${isCurrentMonth ? 'opacity-50 cursor-not-allowed' : ''}`}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                disabled={isCurrentMonth}
              >
                <ChevronRight size={16} className={`text-blue-700 ${isCurrentMonth ? 'text-gray-400' : ''}`} />
              </motion.button>

              {/* Month Picker - hidden but functional */}
              <input
                type="month"
                value={formatMonthForInput(selectedMonthDate)}
                onChange={handleMonthChange}
                max={formatMonthForInput(new Date())}
                className="hidden"
              />

              {/* Reset Button */}
              {!isCurrentMonth && (
                <motion.button
                  onClick={handleResetToCurrentMonth}
                  className="px-3 py-2 bg-gradient-to-r from-green-400 to-green-600 text-white rounded-xl shadow-lg hover:shadow-xl flex items-center gap-1 font-bold text-sm ml-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <CalendarDays size={14} />
                  Current
                </motion.button>
              )}
            </div>
          </div>
          
          <div style={{ width: '100%', height: 300 }} className="relative">
            <ResponsiveContainer>
              <AreaChart data={monthlyChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorPresent" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ccc" opacity={0.5} />
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#4b5563', fontSize: 12 }} type="category" interval={0} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4b5563', fontSize: 12 }} allowDecimals={false}/>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.5)',
                    backdropFilter: 'blur(5px)',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                  }}
                  labelFormatter={(day) => `${currentMonthName.split(' ')[0]} ${day}`}
                />
                <Area type="monotone" dataKey="Present" stroke="#2563eb" strokeWidth={2} fillOpacity={1} fill="url(#colorPresent)" />
              </AreaChart>
            </ResponsiveContainer>
            {monthlyChartData.length === 0 && !isLoading && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
                No data for {currentMonthName}
              </div>
            )}
          </div>
        </motion.div>

        {/* DAILY ACTIVITY HOURS */}
        <motion.div
          className="bg-white/70 backdrop-blur-md shadow-2xl rounded-3xl p-6 border border-white/30 relative"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          whileHover={{ 
            scale: 1.02,
            boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
          }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          {isLoading && (
            <motion.div
              className="absolute inset-0 bg-white/50 backdrop-blur-sm rounded-3xl flex items-center justify-center z-10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            </motion.div>
          )}

          <div className="flex items-center justify-between mb-4">
            <motion.h2 
              className="text-xl font-bold text-blue-800 flex items-center"
              animate={{ 
                color: ["#1e40af", "#7c3aed", "#1e40af"]
              }}
              transition={{ 
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ 
                  duration: 3,
                  repeat: Infinity,
                  ease: "linear"
                }}
              >
                <Clock className="mr-2" size={20} />
              </motion.div>
              Activity Hours
            </motion.h2>

            <div className="flex items-center gap-2">
              <motion.div 
                className="relative"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <input
                  type="date"
                  value={formatDateForInput(selectedDailyDate)}
                  onChange={handleDailyDateChange}
                  max={getMaxDate()}
                  className="px-4 py-2.5 bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border-2 border-purple-300 text-sm font-bold text-purple-800 cursor-pointer hover:bg-purple-50 hover:border-purple-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
                {isToday && (
                  <motion.div 
                    className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-lg"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}
              </motion.div>

              {!isToday && (
                <motion.button
                  onClick={handleResetToToday}
                  className="px-4 py-2.5 bg-gradient-to-r from-green-400 to-green-600 text-white rounded-xl shadow-lg hover:shadow-xl flex items-center gap-2 font-bold text-sm"
                  whileHover={{ scale: 1.05, boxShadow: "0 10px 25px rgba(74, 222, 128, 0.4)" }}
                  whileTap={{ scale: 0.95 }}
                >
                  <RotateCcw size={16} />
                  Today
                </motion.button>
              )}
            </div>
          </div>

          <div style={{ width: '100%', height: 300 }} className="relative">
            <ResponsiveContainer>
              <BarChart data={workingHoursData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity={1}/>
                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0.8}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ccc" opacity={0.5} />
                <XAxis dataKey="hour" axisLine={false} tickLine={false} tick={{ fill: '#4b5563', fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4b5563', fontSize: 12 }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.5)',
                  }}
                />
                <Bar 
                  dataKey="count" 
                  fill="url(#barGradient)" 
                  radius={[8, 8, 0, 0]}
                  animationDuration={1500}
                  animationEasing="ease-out"
                />
              </BarChart>
            </ResponsiveContainer>
            {workingHoursData.length === 0 && !isLoading && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
                No activity for this day
              </div>
            )}
          </div>
        </motion.div>
      </div>

      <motion.div
        className="relative z-10 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6"
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.15 } },
        }}
      >
        <FeatureCard icon={<Users size={40} />} title="Employee Management" description="Add employees with photo & ID, and manage face data with ease." />
        <FeatureCard icon={<Calendar size={40} />} title="Attendance Logs" description="View detailed attendance with timestamps and camera data." />
        <FeatureCard icon={<Camera size={40} />} title="Live Camera Streams" description="Monitor entry and exit feeds in real time." />
        <FeatureCard icon={<Users size={40} />} title="Secure Access" description="Ensures encrypted authentication and reliable system startup." />
      </motion.div>

      <motion.div
        className="relative z-10 text-sm text-gray-600 text-center mt-10"
        animate={{ opacity: [0.8, 1, 0.8] }}
        transition={{ duration: 3, repeat: Infinity }}
      >
        ⏱ Last updated at {lastUpdated}
      </motion.div>
    </motion.div>
  );
}

function AnimatedStatCard({ label, value, color, controls }) {
  const colorMap = {
    green: 'from-green-400/80 to-green-600/80',
    blue: 'from-blue-400/80 to-blue-600/80',
    purple: 'from-purple-400/80 to-purple-600/80',
    red: 'from-red-400/80 to-red-600/80',
  };

  return (
    <motion.div
      className={`p-6 rounded-2xl text-center shadow-lg backdrop-blur-xl bg-gradient-to-br ${colorMap[color]} text-white border border-white/30`}
      whileHover={{ y: -5, scale: 1.05 }}
      animate={controls}
    >
      <motion.div
        className="text-5xl font-bold drop-shadow-lg"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        {value}
      </motion.div>
      <div className="text-sm mt-2 font-semibold opacity-90">{label}</div>
    </motion.div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <motion.div
      className="p-6 bg-white/60 backdrop-blur-md rounded-2xl shadow-lg border border-white/30 hover:shadow-2xl transition-transform relative overflow-hidden"
      whileHover={{ scale: 1.05, y: -4 }}
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0 }
      }}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-transparent via-white/10 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-700"
      />
      <motion.div
        className="text-blue-700 mb-4 drop-shadow-md"
        whileHover={{ rotate: 10, scale: 1.1 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        {icon}
      </motion.div>
      <h3 className="text-lg font-bold text-gray-800 mb-1">{title}</h3>
      <p className="text-gray-700 text-sm">{description}</p>
    </motion.div>
  );
}
