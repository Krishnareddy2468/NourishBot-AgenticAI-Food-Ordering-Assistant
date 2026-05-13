'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import KitchenPage from '../../components/pages/KitchenPage'

export default function KitchenRoute() {
  const { isLoggedIn, role } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoggedIn) router.replace('/login')
    else if (role !== 'KITCHEN' && role !== 'ADMIN') router.replace('/')
  }, [isLoggedIn, role, router])

  if (!isLoggedIn || (role !== 'KITCHEN' && role !== 'ADMIN')) return null
  return <KitchenPage />
}
