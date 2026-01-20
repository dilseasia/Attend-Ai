import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  FiHome,
  FiUsers,
  FiClock,
  FiBookOpen,
  FiCamera,
  FiEye,
  FiClipboard
} from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';

/* -----------------------------
   Sidebar Navigation Items
------------------------------ */
const links = [
  { name: 'Home', path: '/dashboard', icon: <FiHome /> },
  { name: 'Employees', path: '/employees', icon: <FiUsers /> },
  { name: 'Attendance', path: '/attendance', icon: <FiClock /> },
  { name: 'Attendance Request', path: '/attendance-request', icon: <FiClipboard /> },
  { name: 'Report', path: '/log', icon: <FiBookOpen /> },
  { name: 'Live Cameras', path: '/live-cameras', icon: <FiCamera /> },
  { name: 'Anonymous', path: '/anonymous', icon: <FiEye /> },
];

/* -----------------------------
   Animation Variants
------------------------------ */
const sidebarVariants = {
  collapsed: { width: 72 },
  expanded: { width: 260 },
};

const labelVariants = {
  collapsed: {
    opacity: 0,
    x: -20,
    pointerEvents: 'none',
    transition: { duration: 0.15 },
  },
  expanded: {
    opacity: 1,
    x: 0,
    pointerEvents: 'auto',
    transition: { duration: 0.25 },
  },
};

/* -----------------------------
   Sidebar Component
------------------------------ */
export default function Sidebar() {
  const [expanded, setExpanded] = useState(false);
  const location = useLocation();

  return (
    <motion.aside
      className="fixed min-h-screen flex flex-col border-r border-blue-100 shadow-2xl backdrop-blur-md"
      style={{
        background: 'linear-gradient(135deg, #e0e7ff 0%, #f7fafc 100%)',
        top: 0,
        left: 0,
        height: '100vh',
        zIndex: 9999,
      }}
      variants={sidebarVariants}
      animate={expanded ? 'expanded' : 'collapsed'}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      transition={{ type: 'spring', stiffness: 200, damping: 22 }}
    >
      {/* ---------------- Logo ---------------- */}
      <div className="flex items-center justify-center py-7 border-b border-blue-100">
        <motion.div
          className="flex items-center gap-3"
          initial={false}
          animate={expanded ? 'expanded' : 'collapsed'}
        >
          <motion.svg
            xmlns="http://www.w3.org/2000/svg"
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-blue-600 drop-shadow-lg"
            style={{ minWidth: 40 }}
            initial={{ rotate: -10, scale: 0.95 }}
            animate={{ rotate: 0, scale: 1.1 }}
            transition={{ type: 'spring', stiffness: 150 }}
          >
            <path d="M2.5 8.187c.104-2.1.415-3.41 1.347-4.34c.93-.932 2.24-1.243 4.34-1.347M21.5 8.187c-.104-2.1-.415-3.41-1.347-4.34c-.93-.932-2.24-1.243-4.34-1.347m0 19c2.1-.104 3.41-.415 4.34-1.347c.932-.93 1.243-2.24 1.347-4.34M8.187 21.5c-2.1-.104-3.41-.415-4.34-1.347c-.932-.93-1.243-2.24-1.347-4.34M17.5 17l-.202-.849a2 2 0 0 0-1.392-1.458l-2.406-.694v-1.467c.896-.605 1.5-1.736 1.5-3.032C15 7.567 13.656 6 12 6c-1.657 0-3 1.567-3 3.5c0 1.296.603 2.427 1.5 3.032v1.467l-2.391.7a2 2 0 0 0-1.371 1.406L6.5 17" />
          </motion.svg>

          <AnimatePresence>
            {expanded && (
              <motion.div
                className="text-xl font-extrabold text-gray-800 leading-tight tracking-tight"
                variants={labelVariants}
                initial="collapsed"
                animate="expanded"
                exit="collapsed"
              >
                Face Recognition <br /> Attendance System
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* ---------------- Navigation ---------------- */}
      <nav className="flex-1 px-2 py-8 space-y-2 overflow-y-auto">
        {links.map((link) => {
          const isActive = location.pathname === link.path;

          return (
            <NavLink
              key={link.name}
              to={link.path}
              className={`group relative flex items-center px-2 py-3 rounded-2xl font-medium transition-all duration-200 overflow-hidden ${
                isActive
                  ? 'bg-gradient-to-r from-blue-100 to-blue-50 text-blue-700 shadow scale-[1.03]'
                  : 'text-gray-700 hover:bg-blue-50 hover:scale-[1.01]'
              }`}
            >
              {/* Left Active Bar */}
              <span
                className={`absolute left-0 top-0 h-full w-1 rounded-r-2xl transition-all duration-300 ${
                  isActive ? 'bg-blue-500' : 'group-hover:bg-blue-300'
                }`}
              />

              {/* Icon */}
              <motion.span
                className="text-2xl z-10 flex-shrink-0"
                whileHover={{ scale: 1.18, rotate: -8 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                {link.icon}
              </motion.span>

              {/* Label */}
              <AnimatePresence>
                {expanded && (
                  <motion.span
                    className="z-10 ml-4 whitespace-nowrap"
                    variants={labelVariants}
                    initial="collapsed"
                    animate="expanded"
                    exit="collapsed"
                  >
                    {link.name}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          );
        })}
      </nav>

      {/* ---------------- Footer ---------------- */}
      <div className="mt-auto mb-4 px-2">
        <div className="border-t border-blue-100 my-3" />
        <AnimatePresence>
          {expanded && (
            <motion.div
              className="text-xs text-gray-400 text-center py-2"
              variants={labelVariants}
              initial="collapsed"
              animate="expanded"
              exit="collapsed"
            >
              &copy; {new Date().getFullYear()} Smart Attendance
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.aside>
  );
}