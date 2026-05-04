/**
 * Role selector — shown after login if user has multiple role options.
 * For now, every user picks their primary role view.
 */

import { useAuth } from '../../context/AuthContext'
import { LayoutDashboard, ChefHat, UtensilsCrossed } from 'lucide-react'

const ROLES = [
  { id: 'ADMIN', label: 'Admin / POS', icon: LayoutDashboard, desc: 'Orders, menu, staff, payments, drivers' },
  { id: 'WAITER', label: 'Waiter', icon: UtensilsCrossed, desc: 'Table map, sessions, fire-to-kitchen, bill' },
  { id: 'KITCHEN', label: 'Kitchen', icon: ChefHat, desc: 'KDS v2, item-level statuses' },
]

export default function RoleSelector({ onSelect }) {
  const { role, user } = useAuth()

  // If user has a single role, auto-select
  const allowedRoles = ROLES.filter(r => {
    if (role === 'ADMIN') return true
    return r.id === role
  })

  if (allowedRoles.length === 1) {
    // Auto-select after mount
    setTimeout(() => onSelect(allowedRoles[0].id), 0)
    return null
  }

  return (
    <div className="role-selector">
      <h2>Welcome, {user?.email || 'Staff'}</h2>
      <p>Select your view:</p>
      <div className="role-cards">
        {allowedRoles.map(r => {
          const Icon = r.icon
          return (
            <button key={r.id} className="role-card" onClick={() => onSelect(r.id)}>
              <Icon size={32} />
              <span className="role-card-label">{r.label}</span>
              <span className="role-card-desc">{r.desc}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
