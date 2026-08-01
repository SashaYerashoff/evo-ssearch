import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { API_PREFIXES, API_TARGET } from './proxy-config'

// Development talks to Flask on :5000. Production assets are mounted by Flask
// under /ui-assets/, so the appliance never needs Node.js at runtime.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/ui-assets/' : '/',
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
}))
