import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, Trash2, Clock, UserPlus } from "lucide-react";

export default function Employees() {
  const [name, setName] = useState("");
  const [employeeId, setEmployeeId] = useState("");
  const [photo, setPhoto] = useState(null);
  const [preview, setPreview] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [employees, setEmployees] = useState([]);
  const [averages, setAverages] = useState({});
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchEmployees();
    fetchAverageWorkingHours();
  }, []);

  const fetchEmployees = async () => {
    try {
      const res = await fetch("http://10.8.11.183:8000/api/employees");
      const json = await res.json();
      const data = json.employees || json;
      setEmployees(data.sort((a, b) => a.name.localeCompare(b.name)));
    } catch (err) {
      console.error("Failed to load employees", err);
    }
  };

  const fetchAverageWorkingHours = async () => {
    try {
      const res = await fetch("http://10.8.11.183:8000/api/average-working-hours");
      const json = await res.json();
      const avgMap = {};
      json.averages.forEach((a) => {
        avgMap[a.emp_id] = a.avg_hours;
      });
      setAverages(avgMap);
    } catch (err) {
      console.error("Failed to fetch average working hours", err);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!name || !employeeId || !photo) {
      setErrorMessage("Please fill in all fields.");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("emp_id", employeeId);
    formData.append("photo", photo);

    try {
      const res = await fetch("http://10.8.11.183:8000/api/add-employee", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      setName("");
      setEmployeeId("");
      setPhoto(null);
      setPreview(null);
      setErrorMessage("");
      setShowSuccess(true);

      fetchEmployees();
      setTimeout(() => setShowSuccess(false), 2500);
    } catch (error) {
      console.error(error);
      setErrorMessage("Failed to upload. Try again.");
    }
  };

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPhoto(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleDelete = async (emp) => {
    const confirm = window.confirm(`Are you sure you want to delete ${emp.name}?`);
    if (!confirm) return;

    const folderName = `${emp.name}_${emp.emp_id}`;
    try {
      const res = await fetch(`http://10.8.11.183:8000/api/delete-employee/${folderName}`, {
        method: "DELETE",
      });
      
      if (!res.ok) throw new Error("Delete failed");
      
      setEmployees((prev) => prev.filter((e) => e.emp_id !== emp.emp_id));
    } catch (err) {
      alert("Failed to delete employee");
    }
  };

  // Function to get color and styling based on average hours
  const getHoursColorClass = (avgHours) => {
    if (!avgHours) return { textColor: "text-gray-400", bgColor: "bg-gray-100", label: "No data" };
    
    // Parse hours and minutes from format like "8h 44m"
    const hoursMatch = avgHours.match(/(\d+)h/);
    const minutesMatch = avgHours.match(/(\d+)m/);
    
    const hours = hoursMatch ? parseInt(hoursMatch[1]) : 0;
    const minutes = minutesMatch ? parseInt(minutesMatch[1]) : 0;
    const totalHours = hours + minutes / 60;

    if (totalHours >= 9) {
      // Excellent - 9+ hours
      return { 
        textColor: "text-emerald-700", 
        bgColor: "bg-emerald-100", 
        label: avgHours,
        badge: ""
      };
    } else if (totalHours >= 8) {
      // Good - 8-9 hours
      return { 
        textColor: "text-green-700", 
        bgColor: "bg-green-100", 
        label: avgHours,
        badge: ""
      };
    } else if (totalHours >= 7) {
      // Average - 7-8 hours
      return { 
        textColor: "text-yellow-700", 
        bgColor: "bg-yellow-100", 
        label: avgHours,
        badge: ""
      };
    } else if (totalHours >= 6) {
      // Low - 6-7 hours
      return { 
        textColor: "text-orange-700", 
        bgColor: "bg-orange-100", 
        label: avgHours,
        badge: ""
      };
    } else {
      // Very Low - below 6 hours
      return { 
        textColor: "text-red-700", 
        bgColor: "bg-red-100", 
        label: avgHours,
        badge: ""
      };
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen p-10 bg-gradient-to-br from-blue-50 via-white to-indigo-50 text-gray-900"
    >
      {/* Header */}
      <div className="flex items-center mb-10 space-x-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring" }}
          className="p-3 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl shadow-lg text-white"
        >
          <UserPlus size={28} />
        </motion.div>
        <h1 className="text-4xl font-extrabold text-gray-800 tracking-tight">
          Employee Management
        </h1>
      </div>

      {/* Form */}
      <motion.form
        onSubmit={handleUpload}
        className="bg-white/70 backdrop-blur-xl p-8 rounded-3xl shadow-xl border border-white/50 max-w-2xl mb-12"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        <h2 className="text-2xl font-semibold mb-6 text-gray-700">
          Add New Employee
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <label className="block mb-2 font-medium text-gray-600">
              Full Name
            </label>
            <input
              type="text"
              className="w-full px-4 py-3 rounded-2xl border border-gray-200 focus:ring-2 focus:ring-indigo-200 shadow-sm outline-none"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="John Doe"
            />
          </div>
          <div>
            <label className="block mb-2 font-medium text-gray-600">
              Employee ID
            </label>
            <input
              type="text"
              className="w-full px-4 py-3 rounded-2xl border border-gray-200 focus:ring-2 focus:ring-indigo-200 shadow-sm outline-none"
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              placeholder="EMP001"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block mb-2 font-medium text-gray-600">
              Upload Photo
            </label>
            <input
              type="file"
              onChange={handlePhotoChange}
              accept="image/*"
              className="w-full"
            />
            {preview && (
              <img
                src={preview}
                alt="Preview"
                className="w-28 h-28 mt-3 object-cover rounded-full border-4 border-indigo-100 shadow-md"
              />
            )}
          </div>
        </div>

        {errorMessage && (
          <p className="text-red-500 mt-3 font-medium">{errorMessage}</p>
        )}

        <button
          type="submit"
          className="mt-6 bg-gradient-to-r from-indigo-500 to-blue-500 text-white font-semibold px-6 py-3 rounded-full shadow-lg hover:shadow-xl transition-all"
        >
          Upload Employee
        </button>
      </motion.form>

      {/* Success Toast */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ y: -40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -40, opacity: 0 }}
            className="fixed top-8 right-8 bg-white shadow-xl rounded-2xl p-4 flex items-center space-x-3 border border-gray-100 z-50"
          >
            <CheckCircle className="text-green-500" size={24} />
            <span className="font-medium text-gray-700">
              Employee added successfully!
            </span>
          </motion.div>
        )}
      </AnimatePresence>

   

      {/* Search */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-700">All Employees</h2>
        <input
          type="text"
          placeholder="Search by name or ID..."
          className="px-4 py-3 rounded-full border border-gray-300 shadow-sm focus:ring-2 focus:ring-indigo-200 outline-none w-64"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Employee Cards */}
      <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-8">
        {employees
          .filter(
            (emp) =>
              emp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
              emp.emp_id.toLowerCase().includes(searchTerm.toLowerCase())
          )
          .map((emp, i) => {
            const hoursInfo = getHoursColorClass(averages[emp.emp_id]);
            
            return (
              <motion.div
                key={emp.emp_id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-white/60 backdrop-blur-md rounded-3xl shadow-lg border border-gray-100 hover:shadow-2xl transition-all p-6 flex flex-col items-center text-center"
              >
                <img
                  src={emp.image_url}
                  alt={emp.name}
                  className="w-28 h-28 rounded-full object-cover border-4 border-indigo-100 shadow mb-3"
                />
                <h3 className="font-semibold text-lg text-gray-800">
                  {emp.name}
                </h3>
                <p className="text-gray-500 text-sm">ID: {emp.emp_id}</p>
                
                {/* Color-coded average hours */}
                <div className={`flex items-center justify-center text-sm font-semibold mt-3 px-4 py-2 rounded-full ${hoursInfo.bgColor} ${hoursInfo.textColor}`}>
                  <Clock className="mr-2" size={16} />
                  <span>{hoursInfo.label}</span>
                </div>
                
                {hoursInfo.badge && (
                  <span className={`text-xs font-medium mt-2 ${hoursInfo.textColor}`}>
                    {hoursInfo.badge}
                  </span>
                )}

                <button
                  onClick={() => handleDelete(emp)}
                  className="mt-4 px-4 py-2 text-red-600 font-medium bg-red-50 rounded-full hover:bg-red-100 transition flex items-center space-x-1"
                >
                  <Trash2 size={16} />
                  <span>Delete</span>
                </button>
              </motion.div>
            );
          })}
      </div>
    </motion.div>
  );
}