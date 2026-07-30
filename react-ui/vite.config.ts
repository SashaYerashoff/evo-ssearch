import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { API_PREFIXES, API_TARGET } from './proxy-config'

// The React prototype talks to the existing Flask backend running on :5000.
// Everything under these prefixes is proxied there so cookies/CSRF/SSE work.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,        // listen on 0.0.0.0 so the app is reachable from the LAN
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [
        p,
        { target: API_TARGET, changeOrigin: true, ws: true },
      ]),
    ),
  },
})
