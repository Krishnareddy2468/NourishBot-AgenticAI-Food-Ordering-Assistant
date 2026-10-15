'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import AdminPage from '../../components/pages/AdminPage'

export default function AdminRoute() {
  const { isLoggedIn, role, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (role !== 'ADMIN') router.replace('/workspace')
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || !isLoggedIn || role !== 'ADMIN') return null
  return <AdminPage />
}
