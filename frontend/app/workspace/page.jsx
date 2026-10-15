'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import RoleSelector from '../../components/shared/RoleSelector'

export default function WorkspacePage() {
  const { isLoggedIn, authLoading, role } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (!authLoading && role === 'CUSTOMER') router.replace('/account')
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || !isLoggedIn || role === 'CUSTOMER') return null

  function handleRoleSelect(role) {
    const path = role === 'ADMIN' ? '/admin' : role === 'WAITER' ? '/waiter' : '/kitchen'
    router.replace(path)
  }

  return <RoleSelector onSelect={handleRoleSelect} />
}
