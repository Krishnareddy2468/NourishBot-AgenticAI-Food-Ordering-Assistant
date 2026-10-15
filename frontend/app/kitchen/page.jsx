'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import KitchenPage from '../../components/pages/KitchenPage'

export default function KitchenRoute() {
  const { isLoggedIn, role, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (role !== 'KITCHEN' && role !== 'ADMIN') router.replace('/workspace')
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || !isLoggedIn || (role !== 'KITCHEN' && role !== 'ADMIN')) return null
  return <KitchenPage />
}
