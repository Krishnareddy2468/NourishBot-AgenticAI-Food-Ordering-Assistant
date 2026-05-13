'use client'
import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Plus, UserCheck, UserX, Loader2, RefreshCw } from 'lucide-react'
import { fetchStaff, createStaff, toggleStaffActive } from '../../../services/platformApi'

const ROLE_COLORS = {
  ADMIN: { bg: 'rgba(255,107,53,0.15)', color: '#FF6B35' },
  MANAGER: { bg: 'rgba(123,47,190,0.15)', color: '#9D4EDD' },
  CASHIER: { bg: 'rgba(0,78,137,0.2)', color: '#0066CC' },
  WAITER: { bg: 'rgba(59,130,246,0.15)', color: '#3B82F6' },
  KITCHEN: { bg: 'rgba(245,158,11,0.15)', color: '#F59E0B' },
  DRIVER: { bg: 'rgba(26,147,111,0.15)', color: '#1A936F' },
}

const ROLES = ['ADMIN', 'MANAGER', 'CASHIER', 'WAITER', 'KITCHEN', 'DRIVER']

const EMPTY_FORM = { name: '', email: '', phone: '', role: 'WAITER', password: '' }

export default function EmployeesView() {
  const [staff, setStaff] = useState([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState(EMPTY_FORM)
  const [saving, setSaving] = useState(false)
  const [togglingId, setTogglingId] = useState(null)

  const load = useCallback(async () => {
    try {
      const data = await fetchStaff()
      setStaff(data)
    } catch {
      toast.error('Failed to load staff')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function handleCreate(e) {
    e.preventDefault()
    setSaving(true)
    try {
      await createStaff(form)
      toast.success(`${form.name} added as ${form.role}`)
      setForm(EMPTY_FORM)
      setShowForm(false)
      load()
    } catch (err) {
      toast.error(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleToggleActive(member) {
    setTogglingId(member.id)
    try {
      await toggleStaffActive(member.id, !member.active)
      setStaff(prev => prev.map(s => s.id === member.id ? { ...s, active: !s.active } : s))
      toast.success(`${member.name} ${member.active ? 'deactivated' : 'activated'}`)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setTogglingId(null)
    }
  }

  const groupedByRole = ROLES.reduce((acc, role) => {
    const members = staff.filter(s => s.role === role)
    if (members.length > 0) acc[role] = members
    return acc
  }, {})

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading staff...</div>

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>{staff.length} staff members</span>
          <button className="icon-btn" onClick={load}><RefreshCw size={15} /></button>
        </div>
        <button className="btn btn-primary btn-sm" onClick={() => setShowForm(v => !v)}>
          <Plus size={14} /> Add Staff
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="menu-form" style={{ marginBottom: 20 }}>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input className="form-input" required value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))} placeholder="e.g. Ravi Kumar" />
            </div>
            <div className="form-group">
              <label className="form-label">Email</label>
              <input className="form-input" type="email" required value={form.email} onChange={e => setForm(p => ({ ...p, email: e.target.value }))} placeholder="ravi@dzukku.com" />
            </div>
            <div className="form-group">
              <label className="form-label">Phone</label>
              <input className="form-input" value={form.phone} onChange={e => setForm(p => ({ ...p, phone: e.target.value }))} placeholder="9XXXXXXXXX" />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Role</label>
              <select className="form-input" value={form.role} onChange={e => setForm(p => ({ ...p, role: e.target.value }))}>
                {ROLES.map(r => <option key={r}>{r}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <input className="form-input" type="password" required value={form.password} onChange={e => setForm(p => ({ ...p, password: e.target.value }))} placeholder="Temporary password" />
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
              {saving ? <Loader2 size={14} className="spin" /> : 'Save'}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </form>
      )}

      {staff.length === 0 && !showForm && (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 0', fontSize: 14 }}>
          No staff members found. Add your first staff member above.
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {Object.entries(groupedByRole).map(([role, members]) => {
          const rc = ROLE_COLORS[role] || { bg: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)' }
          return (
            <div key={role}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <span style={{ background: rc.bg, color: rc.color, borderRadius: 6, padding: '2px 10px', fontSize: 11, fontWeight: 700 }}>{role}</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{members.length} member{members.length !== 1 ? 's' : ''}</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {members.map(member => (
                  <div
                    key={member.id}
                    className="menu-item-row"
                    style={{ opacity: member.active ? 1 : 0.5 }}
                  >
                    <div className="menu-item-info">
                      <div style={{
                        width: 36, height: 36, borderRadius: '50%',
                        background: rc.bg, color: rc.color,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontWeight: 700, fontSize: 14, flexShrink: 0,
                      }}>
                        {member.name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontWeight: 600 }}>{member.name}</div>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{member.email}</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      {member.phone && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{member.phone}</span>}
                      <span style={{ fontSize: 11, color: member.active ? '#1A936F' : '#6B7280', fontWeight: 600 }}>
                        {member.active ? 'Active' : 'Inactive'}
                      </span>
                      <button
                        className="btn btn-sm btn-ghost"
                        disabled={togglingId === member.id}
                        onClick={() => handleToggleActive(member)}
                        title={member.active ? 'Deactivate' : 'Activate'}
                      >
                        {member.active ? <UserX size={14} /> : <UserCheck size={14} />}
                        {member.active ? 'Deactivate' : 'Activate'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
