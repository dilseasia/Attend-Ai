import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function LiveCameras() {
  const entryRef = useRef(null);
  const exitRef = useRef(null);

  // State for user-added cameras
  const [customCameras, setCustomCameras] = useState([]);
  const [newCamera, setNewCamera] = useState({
    name: '',
    place: '',
    rtsp: '',
  });

  // For fullscreen on custom cameras
  const customRefs = customCameras.map(() => React.createRef());

  const goFullScreen = (ref) => {
    if (ref.current.requestFullscreen) ref.current.requestFullscreen();
    else if (ref.current.webkitRequestFullscreen) ref.current.webkitRequestFullscreen();
    else if (ref.current.msRequestFullscreen) ref.current.msRequestFullscreen();
  };

  const handleAddCamera = (e) => {
    e.preventDefault();
    if (!newCamera.name || !newCamera.place || !newCamera.rtsp) {
      alert('Please fill all fields.');
      return;
    }
    setCustomCameras([...customCameras, { ...newCamera }]);
    setNewCamera({ name: '', place: '', rtsp: '' });
  };

  const handleDeleteCamera = (idx) => {
    setCustomCameras(customCameras.filter((_, i) => i !== idx));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen p-6"
      style={{
        background: 'linear-gradient(135deg, #f0f4ff 0%, #f7fafc 100%)',
        color: '#222',
      }}
    >
      <motion.h2
        className="text-3xl font-bold mb-8 rounded-2xl px-6 py-4 shadow-lg"
        style={{
          background: 'linear-gradient(90deg, #60a5fa 0%, #818cf8 100%)',
          color: '#fff',
        }}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
      >
        Live Camera Streams
      </motion.h2>

      <div className="flex flex-wrap justify-center gap-8">
        <motion.div
          className="text-center bg-white rounded-2xl shadow-lg p-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h3 className="mb-2 font-medium text-lg">Entry Camera</h3>
          <img
            ref={entryRef}
            src="http://10.8.11.183:8000/entry_stream"
            alt="Entry Camera"
            className="w-96 h-64 rounded-2xl border object-contain shadow"
          />
          <div className="mt-2 flex justify-center gap-2">
            <button
              onClick={() => goFullScreen(entryRef)}
              className="bg-blue-600 text-white px-4 py-2 rounded-full shadow hover:bg-blue-700 transition"
            >
              🔳 Fullscreen
            </button>
          </div>
        </motion.div>

        <motion.div
          className="text-center bg-white rounded-2xl shadow-lg p-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <h3 className="mb-2 font-medium text-lg">Exit Camera</h3>
          <img
            ref={exitRef}
            src="http://10.8.11.183:8000/exit_stream"
            alt="Exit Camera"
            className="w-96 h-64 rounded-2xl border object-contain shadow"
          />
          <div className="mt-2 flex justify-center gap-2">
            <button
              onClick={() => goFullScreen(exitRef)}
              className="bg-blue-600 text-white px-4 py-2 rounded-full shadow hover:bg-blue-700 transition"
            >
              🔳 Fullscreen
            </button>
          </div>
        </motion.div>
      </div>

      {/* Show Custom Cameras */}
      <AnimatePresence>
        {customCameras.length > 0 && (
          <motion.div
            className="mt-10 flex flex-wrap justify-center gap-8"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.4 }}
          >
            {customCameras.map((cam, idx) => (
              <motion.div
                key={idx}
                className="text-center relative group bg-white rounded-2xl shadow-lg p-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.3, delay: idx * 0.04 }}
              >
                <h3 className="mb-1 font-medium text-lg">{cam.name}</h3>
                <div className="mb-1 text-gray-500 text-sm">{cam.place}</div>
                <video
                  ref={customRefs[idx]}
                  src={cam.rtsp}
                  controls
                  className="w-96 h-64 rounded-2xl border object-contain shadow"
                  autoPlay
                  muted
                >
                  Your browser does not support the video tag.
                </video>
                <div className="mt-2 flex justify-center gap-2">
                  <button
                    onClick={() => goFullScreen(customRefs[idx])}
                    className="bg-blue-600 text-white px-4 py-2 rounded-full shadow hover:bg-blue-700 transition"
                  >
                    🔳 Fullscreen
                  </button>
                  <button
                    onClick={() => handleDeleteCamera(idx)}
                    className="bg-red-600 text-white px-4 py-2 rounded-full shadow hover:bg-red-700 transition"
                    title="Delete Camera"
                  >
                    🗑 Delete
                  </button>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add Custom Camera Form */}
      <motion.div
        className="mt-12 max-w-xl mx-auto bg-white p-8 rounded-2xl shadow-2xl"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <h3 className="text-lg font-semibold mb-3">Add RTSP Camera Stream</h3>
        <form onSubmit={handleAddCamera} className="flex flex-col gap-3">
          <input
            type="text"
            placeholder="Camera Name"
            className="border px-4 py-3 rounded-full shadow focus:ring-2 focus:ring-blue-200 transition"
            value={newCamera.name}
            onChange={e => setNewCamera({ ...newCamera, name: e.target.value })}
          />
          <input
            type="text"
            placeholder="Place"
            className="border px-4 py-3 rounded-full shadow focus:ring-2 focus:ring-blue-200 transition"
            value={newCamera.place}
            onChange={e => setNewCamera({ ...newCamera, place: e.target.value })}
          />
          <input
            type="text"
            placeholder="RTSP Stream Link (HLS/MJPEG/HTTP recommended)"
            className="border px-4 py-3 rounded-full shadow focus:ring-2 focus:ring-blue-200 transition"
            value={newCamera.rtsp}
            onChange={e => setNewCamera({ ...newCamera, rtsp: e.target.value })}
          />
          <button
            type="submit"
            className="bg-blue-700 text-white px-6 py-3 rounded-full shadow hover:bg-blue-800 transition font-semibold"
          >
            ➕ Add Camera
          </button>
        </form>
        <div className="text-xs text-gray-500 mt-2">
          <b>Note:</b> RTSP links are not natively supported in browsers. Use HLS/MJPEG/HTTP streams or set up a backend to convert RTSP to a browser-compatible format.
        </div>
      </motion.div>
    </motion.div>
  );
}
