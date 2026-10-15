'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import WaiterPage from '../../components/pages/WaiterPage'

export default function WaiterRoute() {
  const { isLoggedIn, role, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (role !== 'WAITER' && role !== 'ADMIN') router.replace('/workspace')
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || !isLoggedIn || (role !== 'WAITER' && role !== 'ADMIN')) return null
  return <WaiterPage />
}
