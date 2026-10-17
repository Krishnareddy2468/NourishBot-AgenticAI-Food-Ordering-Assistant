'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import LoginPage from '../../components/pages/LoginPage'

export default function LoginRoute() {
  const { isLoggedIn, authLoading, role } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && isLoggedIn) {
      router.replace(role === 'CUSTOMER' ? '/account' : '/workspace')
    }
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || isLoggedIn) return null
  return <LoginPage />
}
