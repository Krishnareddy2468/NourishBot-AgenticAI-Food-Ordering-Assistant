'use client'
/* eslint-disable react-refresh/only-export-components */
/**
 * Auth context — manages JWT token, user role, and login/logout.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const AuthContext = createContext(null)

const TOKEN_KEY = 'dzukku_token'
const USER_KEY = 'dzukku_user'

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => {
    if (typeof window === 'undefined') return ''
    return localStorage.getItem(TOKEN_KEY) || ''
  })
  const [user, setUser] = useState(() => {
    if (typeof window === 'undefined') return null
    try {
      return JSON.parse(localStorage.getItem(USER_KEY) || 'null')
    } catch { return null }
  })

  const isLoggedIn = !!token && !!user
  const role = user?.role || ''

  const login = useCallback((tokenValue, userData) => {
    setToken(tokenValue)
    setUser(userData)
    localStorage.setItem(TOKEN_KEY, tokenValue)
    localStorage.setItem(USER_KEY, JSON.stringify(userData))
  }, [])

  const logout = useCallback(() => {
    setToken('')
    setUser(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }, [])

  const authHeaders = token ? { Authorization: `Bearer ${token}` } : {}

  useEffect(() => {
    const handleExpired = () => {
      setToken('')
      setUser(null)
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
    }
    window.addEventListener('dzukku-auth-expired', handleExpired)
    return () => window.removeEventListener('dzukku-auth-expired', handleExpired)
  }, [])

  return (
    <AuthContext.Provider value={{ token, user, role, isLoggedIn, login, logout, authHeaders }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
