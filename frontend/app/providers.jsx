'use client'

import { Toaster } from 'react-hot-toast'
import { AuthProvider } from '../context/AuthContext'

export default function Providers({ children }) {
  return (
    <AuthProvider>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#1A1A28',
            color: '#F0F0FF',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 12,
            fontSize: 13,
            fontFamily: 'Inter, sans-serif',
          },
          success: { iconTheme: { primary: '#1A936F', secondary: '#F0F0FF' } },
          error: { iconTheme: { primary: '#EF4444', secondary: '#F0F0FF' } },
        }}
      />
      {children}
    </AuthProvider>
  )
}
