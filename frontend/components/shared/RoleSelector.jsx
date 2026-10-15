'use client'

import { useEffect } from 'react'
import Link from 'next/link'
import { useAuth } from '../../context/AuthContext'
import {
  ArrowLeft,
  ArrowRight,
  ChefHat,
  LayoutDashboard,
  LogOut,
  ShieldCheck,
  Sparkles,
  UtensilsCrossed,
} from 'lucide-react'

const ROLES = [
  {
    id: 'ADMIN',
    label: 'Canteen Operations',
    shortLabel: 'Operations',
    icon: LayoutDashboard,
    desc: 'Watch upcoming scheduled meals, live orders, menu availability, and service throughput.',
    detail: 'Full service overview',
  },
  {
    id: 'WAITER',
    label: 'Fulfillment Desk',
    shortLabel: 'Fulfillment',
    icon: UtensilsCrossed,
    desc: 'Coordinate counter pickup, drive-through lanes, driver dispatch, and verified handoff.',
    detail: 'Collection and delivery',
  },
  {
    id: 'KITCHEN',
    label: 'Kitchen Display',
    shortLabel: 'Kitchen',
    icon: ChefHat,
    desc: 'Manage the preparation queue and move every line item toward readiness.',
    detail: 'Live preparation queue',
  },
]

export default function RoleSelector({ onSelect }) {
  const { role, user, logout } = useAuth()

  const allowedRoles = ROLES.filter(option => {
    if (role === 'ADMIN') return true
    return option.id === role
  })

  useEffect(() => {
    if (allowedRoles.length === 1) {
      onSelect(allowedRoles[0].id)
    }
  }, [allowedRoles, onSelect])

  if (allowedRoles.length === 1) return null

  return (
    <main className="staff-workspace-page">
      <header className="staff-auth-nav">
        <Link className="staff-auth-brand" href="/" aria-label="NourishBot home">
          <span><UtensilsCrossed size={19} /></span>
          NourishBot
        </Link>
        <div className="staff-workspace-nav-actions">
          <span className="staff-workspace-user"><ShieldCheck size={13} /> {user?.email || 'Staff account'}</span>
          <button type="button" onClick={logout}><LogOut size={14} /> Sign out</button>
        </div>
      </header>

      <section className="staff-workspace-shell">
        <div className="staff-workspace-heading">
          <div className="staff-auth-eyebrow"><Sparkles size={14} /> Scheduled-meal coordination</div>
          <h1>Where are you working today?</h1>
          <p>Choose a workspace. Every view stays connected to the same verified order queue.</p>
        </div>

        <div className="staff-workspace-grid">
          {allowedRoles.map((option, index) => {
            const Icon = option.icon
            return (
              <button
                key={option.id}
                type="button"
                className={`staff-workspace-card staff-workspace-card-${index + 1}`}
                onClick={() => onSelect(option.id)}
              >
                <span className="staff-workspace-card-number">0{index + 1}</span>
                <span className="staff-workspace-card-icon"><Icon size={25} /></span>
                <span className="staff-workspace-card-kicker">{option.detail}</span>
                <strong>{option.label}</strong>
                <span className="staff-workspace-card-copy">{option.desc}</span>
                <span className="staff-workspace-card-cta">Open {option.shortLabel} <ArrowRight size={15} /></span>
              </button>
            )
          })}
        </div>

        <div className="staff-workspace-footnote">
          <Link href="/"><ArrowLeft size={14} /> Return to public website</Link>
          <span>Role access is controlled by your authenticated staff account.</span>
        </div>
      </section>
    </main>
  )
}
