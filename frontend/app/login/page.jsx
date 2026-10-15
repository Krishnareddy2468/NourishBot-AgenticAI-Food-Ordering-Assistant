'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import LoginPage from '../../components/pages/LoginPage'

export default function LoginRoute() {
  const { isLoggedIn, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && isLoggedIn) router.replace('/workspace')
  }, [authLoading, isLoggedIn, router])

  if (authLoading || isLoggedIn) return null
  return <LoginPage />
}
