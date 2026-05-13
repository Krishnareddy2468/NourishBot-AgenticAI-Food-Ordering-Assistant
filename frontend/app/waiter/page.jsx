'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import WaiterPage from '../../components/pages/WaiterPage'

export default function WaiterRoute() {
  const { isLoggedIn, role } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoggedIn) router.replace('/login')
    else if (role !== 'WAITER' && role !== 'ADMIN') router.replace('/')
  }, [isLoggedIn, role, router])

  if (!isLoggedIn || (role !== 'WAITER' && role !== 'ADMIN')) return null
  return <WaiterPage />
}
