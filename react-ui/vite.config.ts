import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The React prototype talks to the existing Flask backend running on :5000.
// Everything under these prefixes is proxied there so cookies/CSRF/SSE work.
const API_TARGET = 'http://127.0.0.1:5000'
const API_PREFIXES = [
  '/auth', '/detections', '/luxriot', '/probes', '/agent',
  '/describe_image', '/lm', '/settings', '/health', '/ready',
  '/branding', '/image', '/comments', '/commented_images',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [
        p,
        { target: API_TARGET, changeOrigin: true, ws: true },
      ]),
    ),
  },
})
