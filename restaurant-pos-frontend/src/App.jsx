/**
 * Dzukku POS — Main App with auth, role-based routing, and realtime.
 *
 * Structure:
 *   /login      — Login page
 *   /admin/*    — Admin/POS portal
 *   /waiter/*   — Waiter table service
 *   /kitchen/*  — Kitchen KDS v2
 */

import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { AuthProvider, useAuth } from './context/AuthContext'
import LoginPage from './pages/auth/LoginPage'
import AdminPage from './pages/admin/AdminPage'
import WaiterPage from './pages/waiter/WaiterPage'
import KitchenPage from './pages/kitchen/KitchenPage'
import TrackingPage from './pages/tracking/TrackingPage'
import RoleSelector from './components/shared/RoleSelector'
import './index.css'

function ProtectedRoute({ children, allowedRoles }) {
  const { isLoggedIn, role } = useAuth()
  if (!isLoggedIn) return <Navigate to="/login" replace />
  if (allowedRoles && !allowedRoles.includes(role) && role !== 'ADMIN') {
    return <Navigate to="/" replace />
  }
  return children
}

function AppRoutes() {
  const { isLoggedIn } = useAuth()
  const navigate = useNavigate()

  function handleRoleSelect(role) {
    const path = role === 'ADMIN' ? '/admin' : role === 'WAITER' ? '/waiter' : '/kitchen'
    navigate(path, { replace: true })
  }

  return (
    <Routes>
      <Route path="/login" element={isLoggedIn ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route path="/" element={
        isLoggedIn
          ? <RoleSelector onSelect={handleRoleSelect} />
          : <Navigate to="/login" replace />
      } />
      <Route path="/admin/*" element={<ProtectedRoute><AdminPage /></ProtectedRoute>} />
      <Route path="/waiter/*" element={<ProtectedRoute allowedRoles={['WAITER', 'ADMIN']}><WaiterPage /></ProtectedRoute>} />
      <Route path="/kitchen/*" element={<ProtectedRoute allowedRoles={['KITCHEN', 'ADMIN']}><KitchenPage /></ProtectedRoute>} />
      <Route path="/track/:orderRef" element={<TrackingPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: '#1A1A28', color: '#F0F0FF',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 12, fontSize: 13, fontFamily: 'Inter, sans-serif',
            },
            success: { iconTheme: { primary: '#1A936F', secondary: '#F0F0FF' } },
            error: { iconTheme: { primary: '#EF4444', secondary: '#F0F0FF' } },
          }}
        />
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}
