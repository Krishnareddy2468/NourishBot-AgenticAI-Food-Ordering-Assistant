'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import LoginPage from '../../components/pages/LoginPage'

export default function LoginRoute() {
  const { isLoggedIn } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (isLoggedIn) router.replace('/')
  }, [isLoggedIn, router])

  if (isLoggedIn) return null
  return <LoginPage />
}
