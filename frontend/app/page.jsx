'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../context/AuthContext'
import RoleSelector from '../components/shared/RoleSelector'

export default function HomePage() {
  const { isLoggedIn } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoggedIn) router.replace('/login')
  }, [isLoggedIn, router])

  if (!isLoggedIn) return null

  function handleRoleSelect(role) {
    const path = role === 'ADMIN' ? '/admin' : role === 'WAITER' ? '/waiter' : '/kitchen'
    router.replace(path)
  }

  return <RoleSelector onSelect={handleRoleSelect} />
}
