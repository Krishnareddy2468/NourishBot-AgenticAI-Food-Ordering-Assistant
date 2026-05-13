'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import AdminPage from '../../components/pages/AdminPage'

export default function AdminRoute() {
  const { isLoggedIn, role } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoggedIn) router.replace('/login')
    else if (role !== 'ADMIN') router.replace('/')
  }, [isLoggedIn, role, router])

  if (!isLoggedIn || role !== 'ADMIN') return null
  return <AdminPage />
}
