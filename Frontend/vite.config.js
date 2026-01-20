import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // or you can explicitly set your IP, e.g., '192.168.1.100'
    port: 5173
  }
});
