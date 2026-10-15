'use client'
/**
 * Role selector — shown after login if user has multiple role options.
 * For now, every user picks their primary role view.
 */

import { useEffect } from 'react'
import { useAuth } from '../../context/AuthContext'
import { LayoutDashboard, ChefHat, UtensilsCrossed, Sparkles, ArrowRight } from 'lucide-react'

const ROLES = [
  { id: 'ADMIN', label: 'Canteen Operations', icon: LayoutDashboard, desc: 'Pre-orders, menu availability, team, and throughput' },
  { id: 'WAITER', label: 'Pickup Desk', icon: UtensilsCrossed, desc: 'Customer handoff, active sessions, and collection status' },
  { id: 'KITCHEN', label: 'Kitchen', icon: ChefHat, desc: 'Preparation queue and item-level progress' },
]

export default function RoleSelector({ onSelect }) {
  const { role, user } = useAuth()

  // If user has a single role, auto-select
  const allowedRoles = ROLES.filter(r => {
    if (role === 'ADMIN') return true
    return r.id === role
  })

  useEffect(() => {
    if (allowedRoles.length === 1) {
      onSelect(allowedRoles[0].id)
    }
  }, [allowedRoles, onSelect])

  if (allowedRoles.length === 1) return null

  return (
    <div className="role-selector">
      <div className="role-selector-shell">
        <div className="role-selector-copy">
          <span className="role-chip"><Sparkles size={14} /> Live canteen coordination</span>
          <h2>Choose your workspace for this shift</h2>
          <p>{user?.email || 'Staff'} can move between pickup, kitchen, and operations without losing sync.</p>
        </div>
      <div className="role-cards">
        {allowedRoles.map(r => {
          const Icon = r.icon
          return (
            <button key={r.id} className="role-card" onClick={() => onSelect(r.id)}>
              <div className="role-card-icon">
                <Icon size={30} />
              </div>
              <span className="role-card-label">{r.label}</span>
              <span className="role-card-desc">{r.desc}</span>
              <span className="role-card-cta">Open workspace <ArrowRight size={14} /></span>
            </button>
          )
        })}
      </div>
      </div>
    </div>
  )
}
