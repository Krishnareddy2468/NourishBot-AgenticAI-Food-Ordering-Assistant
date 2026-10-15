'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import RoleSelector from '../../components/shared/RoleSelector'

export default function WorkspacePage() {
  const { isLoggedIn, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
  }, [authLoading, isLoggedIn, router])

  if (authLoading || !isLoggedIn) return null

  function handleRoleSelect(role) {
    const path = role === 'ADMIN' ? '/admin' : role === 'WAITER' ? '/waiter' : '/kitchen'
    router.replace(path)
  }

  return <RoleSelector onSelect={handleRoleSelect} />
}
