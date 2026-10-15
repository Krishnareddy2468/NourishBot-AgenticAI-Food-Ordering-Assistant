'use client'
/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { getCurrentUser, logoutApi } from '../services/platformApi'

const AuthContext = createContext(null)

function normalizeUser(data) {
  if (!data) return null
  return {
    id: data.user_id ?? data.id,
    email: data.email || '',
    role: data.role || '',
    restaurant_id: data.restaurant_id,
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)

  const refreshSession = useCallback(async () => {
    try {
      setUser(normalizeUser(await getCurrentUser()))
    } catch {
      setUser(null)
    } finally {
      setAuthLoading(false)
    }
  }, [])

  useEffect(() => { refreshSession() }, [refreshSession])

  const login = useCallback((userData) => {
    setUser(normalizeUser(userData))
    setAuthLoading(false)
  }, [])

  const logout = useCallback(async () => {
    try { await logoutApi() } catch { /* clear local state even if the request fails */ }
    setUser(null)
  }, [])

  useEffect(() => {
    const handleExpired = () => setUser(null)
    window.addEventListener('nourishbot-auth-expired', handleExpired)
    return () => window.removeEventListener('nourishbot-auth-expired', handleExpired)
  }, [])

  const isLoggedIn = !!user
  const role = user?.role || ''

  return (
    <AuthContext.Provider value={{ user, role, isLoggedIn, authLoading, login, logout, refreshSession }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
