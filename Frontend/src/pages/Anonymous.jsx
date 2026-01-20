import React, { useEffect, useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiTrash2, FiCheckCircle, FiUserCheck } from "react-icons/fi";
import axios from "axios";

export default function Anonymous() {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState(
    () => new Date().toISOString().split("T")[0]
  );
  const [images, setImages] = useState([]);
  const [filter, setFilter] = useState("all");
  const [vehicleType, setVehicleType] = useState("all");
  const [fromHour, setFromHour] = useState("");
  const [toHour, setToHour] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);

  const [modalOpen, setModalOpen] = useState(false);
  const [current, setCurrent] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [employees, setEmployees] = useState([]);
  const [convertModal, setConvertModal] = useState(false);
  const [showConvertSuccess, setShowConvertSuccess] = useState(false);

  const observer = useRef();

  function formatPhotoTime(filePath) {
    const filename = filePath.split("/").pop().replace(".jpg", "");
    const clean = filename.split("?ts=")[0];
    const match = clean.match(/(\d{2})[-_]?(\d{2})[-_]?(\d{2})/);
    if (match) {
      const [_, h, m, s] = match;
      return `${h}:${m}:${s}`;
    }
    return filename;
  }

  useEffect(() => {
    axios
      .get("http://10.8.21.51:8000/api/anonymous-dates")
      .then((res) => setDates(res.data.dates || []))
      .catch(() => setDates([]));

    axios
      .get("http://10.8.21.51:8000/api/employees")
      .then((res) => {
        const data = Array.isArray(res.data)
          ? res.data
          : res.data.employees || [];
        setEmployees(data);
      })
      .catch(() => setEmployees([]));
  }, []);

  const fetchImages = useCallback(async () => {
    if (!selectedDate || loading || !hasMore) return;
    setLoading(true);
    try {
      const res = await axios.get("http://10.8.21.51:8000/api/anonymous-images", {
        params: {
          date: selectedDate,
          filter,
          from_hour: fromHour || undefined,
          to_hour: toHour || undefined,
          vehicle_type: vehicleType,
          page,
          limit: 50,
        },
      });
      const newImages = res.data.images || [];
      setImages((prev) => (page === 1 ? newImages : [...prev, ...newImages]));
      setHasMore(newImages.length > 0);
    } catch (err) {
      console.error("❌ Failed to load images:", err);
    } finally {
      setLoading(false);
    }
  }, [selectedDate, filter, fromHour, toHour, vehicleType, page, loading, hasMore]);

  useEffect(() => {
    fetchImages();
  }, [page, selectedDate, filter, fromHour, toHour, vehicleType]);

  useEffect(() => {
    setImages([]);
    setPage(1);
    setHasMore(true);
  }, [selectedDate, filter, fromHour, toHour, vehicleType]);

  const lastImageRef = useCallback(
    (node) => {
      if (loading) return;
      if (observer.current) observer.current.disconnect();
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore && !loading) {
          setPage((prev) => prev + 1);
        }
      });
      if (node) observer.current.observe(node);
    },
    [loading, hasMore]
  );

  const openModal = (idx) => {
    setCurrent(idx);
    setModalOpen(true);
  };
  const closeModal = () => setModalOpen(false);

  const handleDelete = async () => {
    if (!images[current]) return;
    setDeleting(true);
    try {
      let imgPath = images[current].path.split("?")[0];
      if (imgPath.startsWith("/")) imgPath = imgPath.slice(1);
      await axios.post("http://10.8.21.51:8000/api/delete-anonymous-image", {
        path: imgPath,
      });
      setImages((prev) => prev.filter((_, idx) => idx !== current));
      setModalOpen(false);
    } catch (err) {
      alert("Failed to delete image.");
    }
    setDeleting(false);
  };

  const handleConvertSubmit = async (e) => {
    e.preventDefault();
    const emp_id = e.target.emp_id.value;
    const camera = e.target.camera.value;
    if (!emp_id || !camera) return alert("Please select both fields.");

    try {
      const emp = employees.find((e) => e.emp_id === emp_id);
      const name = emp?.name || "Unknown";
      let anon_path = images[current].path.split("?")[0];
      if (anon_path.startsWith("/")) anon_path = anon_path.slice(1);

      const res = await axios.post(
        "http://10.8.21.51:8000/api/convert-anonymous",
        { emp_id, name, anon_path, camera }
      );

      if (res.data?.success) {
        setConvertModal(false);
        setShowConvertSuccess(true);
        setTimeout(() => setShowConvertSuccess(false), 2000);
        setImages((prev) => prev.filter((_, i) => i !== current));
        setModalOpen(false);
      } else alert(res.data.error || "Conversion failed");
    } catch (err) {
      console.error(err);
      alert("Conversion failed.");
    }
  };

  return (
    <motion.div className="min-h-screen p-6 bg-gradient-to-br from-blue-50 to-gray-50 text-gray-900">
      <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-blue-400 text-white p-4 rounded-2xl shadow-lg">
        Anonymous Detections
      </h2>

      {/* Filters */}
      <form className="mb-8 flex flex-wrap gap-4 items-center">
        <label>Date:</label>
        <input
          type="date"
          className="border rounded-full px-4 py-2"
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
        />

        <label>Filter:</label>
        <select
          className="border rounded-full px-4 py-2"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">All</option>
          <option value="entry">Entry</option>
          <option value="exit">Exit</option>
        </select>

        <label>Vehicle:</label>
        <select
          className="border rounded-full px-4 py-2"
          value={vehicleType}
          onChange={(e) => setVehicleType(e.target.value)}
        >
          <option value="all">All</option>
          <option value="car">Car</option>
          <option value="truck">Truck</option>
          <option value="unknown">Unknown</option>
        </select>

        <label>From:</label>
        <input
          type="time"
          className="border rounded-full px-4 py-2"
          value={fromHour}
          onChange={(e) => setFromHour(e.target.value)}
        />
        <label>To:</label>
        <input
          type="time"
          className="border rounded-full px-4 py-2"
          value={toHour}
          onChange={(e) => setToHour(e.target.value)}
        />
      </form>

      {/* Image Grid */}
      {images.length === 0 && !loading ? (
        <div className="text-center text-gray-500">
          No anonymous detections found.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
            {images.map((img, idx) => {
              const vehicle = img.vehicle_type || "unknown";
              const highlight =
                vehicle === "car" ? "border-2 border-yellow-400" : "";

              const refProp = idx === images.length - 1 ? { ref: lastImageRef } : {};

              return (
                <div
                  key={img.path}
                  {...refProp}
                  className={`relative text-center rounded-2xl overflow-hidden bg-white ${highlight}`}
                >
                  <motion.img
                    src={`http://10.8.21.51:8000${img.path}`}
                    onClick={() => openModal(idx)}
                    className="rounded-2xl w-full h-48 object-cover cursor-pointer hover:scale-105 transition"
                    whileHover={{ scale: 1.05 }}
                  />
                  {vehicle === "car" && (
                    <span className="absolute top-2 left-2 backdrop-blur-md text-black text-xs font-semibold px-3 py-1 rounded-full shadow-sm bg-yellow-300/80">
                      🚗 Car
                    </span>
                  )}
                  <div className="text-xs text-gray-600 mt-1">
                    {formatPhotoTime(img.path)} • {img.type}
                  </div>
                </div>
              );
            })}
          </div>

          {loading && (
            <div className="text-center text-gray-500 mt-4">Loading more...</div>
          )}
        </>
      )}

      {/* Image Modal */}
      <AnimatePresence>
        {modalOpen && images[current] && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="relative bg-white rounded-2xl shadow-2xl p-6"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
            >
              {images[current].vehicle_type === "car" && (
                <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-yellow-300 text-black px-4 py-1 rounded-full text-sm font-bold shadow">
                  🚗 Car Detected
                </div>
              )}

              <button
                onClick={() => setConvertModal(true)}
                className="absolute top-4 left-4 bg-green-600 text-white rounded-full p-2 shadow hover:bg-green-700"
                title="Mark Attendance"
              >
                <FiUserCheck />
              </button>

              <button
                onClick={handleDelete}
                disabled={deleting}
                className="absolute top-4 right-4 bg-red-500 text-white rounded-full p-2 shadow hover:bg-red-600"
                title="Delete"
              >
                <FiTrash2 />
              </button>

              <img
                src={`http://10.8.21.51:8000${images[current].path}`}
                className="rounded-2xl w-[80vw] max-w-xl h-[60vh] object-contain"
              />

              <div className="text-center mt-3 text-gray-600">
                {formatPhotoTime(images[current].path)} ({images[current].type})
              </div>

              <button
                onClick={closeModal}
                className="absolute bottom-4 right-4 bg-gray-200 px-4 py-2 rounded-full hover:bg-gray-300"
              >
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Convert Modal */}
      {convertModal && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <motion.form
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md space-y-5"
            onSubmit={handleConvertSubmit}
          >
            <h3 className="text-xl font-bold mb-2 text-blue-700">
              Mark Attendance
            </h3>
            <div>
              <label className="block mb-1 font-medium">Employee</label>
              <select
                name="emp_id"
                className="border px-4 py-3 rounded-full w-full"
                required
              >
                <option value="">Select Employee</option>
                {employees.map((emp) => (
                  <option key={emp.emp_id} value={emp.emp_id}>
                    {emp.name} ({emp.emp_id})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block mb-1 font-medium">Camera</label>
              <select
                name="camera"
                className="border px-4 py-3 rounded-full w-full"
                required
              >
                <option value="Entry">Entry</option>
                <option value="Exit">Exit</option>
              </select>
            </div>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConvertModal(false)}
                className="px-5 py-2 rounded-full bg-gray-200 hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-5 py-2 rounded-full bg-green-600 text-white hover:bg-green-700"
              >
                Convert
              </button>
            </div>
          </motion.form>
        </div>
      )}

      {/* ✅ Success Toast */}
      <AnimatePresence>
        {showConvertSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-30 z-50"
          >
            <div className="bg-white rounded-2xl shadow-2xl p-8 flex flex-col items-center border">
              <FiCheckCircle className="text-green-500 text-5xl mb-2" />
              <div className="text-lg font-semibold">
                Conversion Successful
              </div>
              <div className="text-gray-600 text-sm">
                Face moved and attendance logged.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
