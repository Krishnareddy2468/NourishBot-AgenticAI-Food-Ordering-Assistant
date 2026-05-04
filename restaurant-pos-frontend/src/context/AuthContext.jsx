/**
 * Auth context — manages JWT token, user role, and login/logout.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const AuthContext = createContext(null)

const TOKEN_KEY = 'dzukku_token'
const USER_KEY = 'dzukku_user'

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || '')
  const [user, setUser] = useState(() => {
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

  // Clear user data when token is removed externally
  useEffect(() => {
    if (!token && user) {
      setUser(null)
      localStorage.removeItem(USER_KEY)
    }
  }, [token])

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
