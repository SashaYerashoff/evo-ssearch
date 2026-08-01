import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { AppearanceProvider } from './appearance/AppearanceProvider'
import './styles/theme.css'
import './styles/app.css'
import './styles/appearance.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppearanceProvider>
      <App />
    </AppearanceProvider>
  </React.StrictMode>,
)
