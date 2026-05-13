'use client'
import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Users, RefreshCw, Loader2, Plus, Trash2, AlertTriangle, Edit2, Check, X } from 'lucide-react'
import { fetchTables, fetchActiveSessions, createTable, updateTable, deleteTable } from '../../../services/platformApi'
import { useWebSocket } from '../../../hooks/useWebSocket'
import { useAuth } from '../../../context/AuthContext'

const STATUS_COLOR = {
  AVAILABLE: '#1A936F',
  OCCUPIED: '#FF6B35',
  RESERVED: '#8B5CF6',
  INACTIVE: '#6B7280',
}

const STATUS_BG = {
  AVAILABLE: 'rgba(26,147,111,0.12)',
  OCCUPIED: 'rgba(255,107,53,0.12)',
  RESERVED: 'rgba(139,92,246,0.12)',
  INACTIVE: 'rgba(107,114,128,0.12)',
}

const EMPTY_FORM = { name: '', capacity: 4 }

export default function TablesView() {
  const { user } = useAuth()
  const [tables, setTables] = useState([])
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState(EMPTY_FORM)
  const [saving, setSaving] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [editForm, setEditForm] = useState({})
  const [deletingId, setDeletingId] = useState(null)
  const { on } = useWebSocket(user?.restaurant_id || 1)

  const load = useCallback(async (silent = false) => {
    if (!silent) setError(null)
    try {
      const [t, s] = await Promise.all([fetchTables(), fetchActiveSessions()])
      setTables(t)
      setSessions(s)
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to connect to backend')
      if (!silent) toast.error('Failed to load tables — is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    const handler = evt => {
      if (evt.event_type?.startsWith('table_session')) load(true)
    }
    return on('*', handler)
  }, [on, load])

  async function handleCreate(e) {
    e.preventDefault()
    if (!form.name.trim()) { toast.error('Table name is required'); return }
    setSaving(true)
    try {
      const created = await createTable({ name: form.name.trim(), capacity: Number(form.capacity) })
      setTables(prev => [...prev, created])
      setForm(EMPTY_FORM)
      setShowForm(false)
      toast.success(`Table "${created.name}" created`)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleUpdate(tableId) {
    try {
      await updateTable(tableId, { name: editForm.name, capacity: Number(editForm.capacity) })
      setTables(prev => prev.map(t => t.id === tableId ? { ...t, ...editForm } : t))
      setEditingId(null)
      toast.success('Table updated')
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleDelete(tableId, tableName) {
    if (!window.confirm(`Delete table "${tableName}"? This cannot be undone.`)) return
    setDeletingId(tableId)
    try {
      await deleteTable(tableId)
      setTables(prev => prev.filter(t => t.id !== tableId))
      toast.success(`Table "${tableName}" deleted`)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setDeletingId(null)
    }
  }

  if (loading) return (
    <div className="page-loading">
      <Loader2 className="spin" size={24} />Loading tables...
    </div>
  )

  if (error) return (
    <div style={{ padding: 24 }}>
      <div style={{
        background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
        borderRadius: 12, padding: '20px 24px', display: 'flex', alignItems: 'flex-start', gap: 14,
      }}>
        <AlertTriangle size={20} style={{ color: '#EF4444', flexShrink: 0, marginTop: 2 }} />
        <div>
          <div style={{ fontWeight: 700, color: '#EF4444', marginBottom: 4 }}>Failed to load tables</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>{error}</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Make sure the backend is running: <code style={{ background: 'var(--bg-overlay)', padding: '1px 6px', borderRadius: 4 }}>cd DzukkuBot && python main.py</code>
          </div>
          <button className="btn btn-primary btn-sm" style={{ marginTop: 14 }} onClick={() => load()}>
            <RefreshCw size={13} /> Retry
          </button>
        </div>
      </div>
    </div>
  )

  const openSessions = sessions.filter(s => s.status === 'OPEN')
  const enriched = tables.map(t => {
    const session = openSessions.find(s => s.table_id === t.id)
    const displayStatus = session ? 'OCCUPIED' : (t.active ? 'AVAILABLE' : 'INACTIVE')
    return { ...t, session, displayStatus }
  })

  const counts = {
    AVAILABLE: enriched.filter(t => t.displayStatus === 'AVAILABLE').length,
    OCCUPIED: enriched.filter(t => t.displayStatus === 'OCCUPIED').length,
    INACTIVE: enriched.filter(t => t.displayStatus === 'INACTIVE').length,
  }

  return (
    <div style={{ padding: 16 }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {Object.entries(counts).map(([status, count]) => (
            <div key={status} className="stat-card" style={{ minWidth: 110, padding: '12px 16px' }}>
              <div className="stat-label">{status}</div>
              <div className="stat-value" style={{ fontSize: 22, color: STATUS_COLOR[status] }}>{count}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="icon-btn" onClick={() => load()} title="Refresh"><RefreshCw size={15} /></button>
          <button className="btn btn-primary btn-sm" onClick={() => setShowForm(v => !v)}>
            <Plus size={14} /> Add Table
          </button>
        </div>
      </div>

      {/* Add table form */}
      {showForm && (
        <form onSubmit={handleCreate} style={{
          background: 'var(--bg-overlay)', border: '1px solid var(--border)',
          borderRadius: 12, padding: 16, marginBottom: 20,
          display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap',
        }}>
          <div className="form-group" style={{ flex: '1 1 160px' }}>
            <label className="form-label">Table Name / Number</label>
            <input
              className="form-input"
              placeholder="e.g. T1 or Window Seat"
              value={form.name}
              onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
              required
              autoFocus
            />
          </div>
          <div className="form-group" style={{ flex: '0 0 120px' }}>
            <label className="form-label">Capacity (seats)</label>
            <input
              className="form-input"
              type="number"
              min={1}
              max={50}
              value={form.capacity}
              onChange={e => setForm(p => ({ ...p, capacity: e.target.value }))}
              required
            />
          </div>
          <div style={{ display: 'flex', gap: 8, paddingBottom: 1 }}>
            <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
              {saving ? <Loader2 size={13} className="spin" /> : <Check size={13} />}
              {saving ? 'Saving...' : 'Save'}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setShowForm(false); setForm(EMPTY_FORM) }}>
              <X size={13} /> Cancel
            </button>
          </div>
        </form>
      )}

      {/* Empty state */}
      {tables.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '60px 24px',
          background: 'var(--bg-card)', border: '1px dashed var(--border)', borderRadius: 14,
        }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🪑</div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>No tables yet</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 16 }}>
            Add your restaurant's tables to enable table management and waiter workflows.
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => setShowForm(true)}>
            <Plus size={14} /> Add Your First Table
          </button>
        </div>
      )}

      {/* Table grid */}
      {tables.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 14, marginBottom: 28 }}>
          {enriched.map(t => {
            const color = STATUS_COLOR[t.displayStatus]
            const bg = STATUS_BG[t.displayStatus]
            const isEditing = editingId === t.id

            return (
              <div
                key={t.id}
                style={{
                  background: bg, border: `2px solid ${color}`, borderRadius: 14,
                  padding: 14, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', gap: 6, minHeight: 120,
                  justifyContent: 'center', position: 'relative',
                }}
              >
                {isEditing ? (
                  <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <input
                      className="form-input"
                      style={{ textAlign: 'center', padding: '4px 8px', fontSize: 13 }}
                      value={editForm.name}
                      onChange={e => setEditForm(p => ({ ...p, name: e.target.value }))}
                      autoFocus
                    />
                    <input
                      className="form-input"
                      type="number" min={1}
                      style={{ textAlign: 'center', padding: '4px 8px', fontSize: 13 }}
                      value={editForm.capacity}
                      onChange={e => setEditForm(p => ({ ...p, capacity: e.target.value }))}
                    />
                    <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                      <button className="btn btn-sm btn-primary" style={{ padding: '3px 8px' }} onClick={() => handleUpdate(t.id)}><Check size={12} /></button>
                      <button className="btn btn-sm btn-ghost" style={{ padding: '3px 8px' }} onClick={() => setEditingId(null)}><X size={12} /></button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                      {t.table_number || t.name}
                    </div>
                    <div style={{ fontSize: 11, fontWeight: 600, color }}>{t.displayStatus}</div>
                    {t.session && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                        <Users size={11} />{t.session.guests} guests
                      </div>
                    )}
                    {t.capacity && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Seats {t.capacity}</div>
                    )}
                    {/* Edit/Delete */}
                    <div style={{
                      position: 'absolute', top: 6, right: 6,
                      display: 'flex', gap: 2, opacity: 0,
                      transition: 'opacity 0.15s',
                    }}
                      className="table-actions"
                    >
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '2px 4px', fontSize: 10 }}
                        onClick={() => { setEditingId(t.id); setEditForm({ name: t.name, capacity: t.capacity }) }}
                      >
                        <Edit2 size={11} />
                      </button>
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '2px 4px', color: '#EF4444' }}
                        disabled={deletingId === t.id}
                        onClick={() => handleDelete(t.id, t.name)}
                      >
                        {deletingId === t.id ? <Loader2 size={11} className="spin" /> : <Trash2 size={11} />}
                      </button>
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Active sessions */}
      {openSessions.length > 0 && (
        <div>
          <div className="section-header">
            <div className="section-title">Active Sessions</div>
          </div>
          <div className="orders-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Table</th>
                  <th>Guests</th>
                  <th>Opened At</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {openSessions.map(s => {
                  const table = tables.find(t => t.id === s.table_id)
                  return (
                    <tr key={s.id}>
                      <td style={{ fontFamily: 'monospace', color: 'var(--brand-primary)', fontWeight: 700 }}>#{s.id}</td>
                      <td style={{ fontWeight: 600 }}>{table?.table_number || table?.name || `Table ${s.table_id}`}</td>
                      <td><div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Users size={13} />{s.guests}</div></td>
                      <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {s.opened_at ? new Date(s.opened_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '—'}
                      </td>
                      <td><span className="badge badge-ready">OPEN</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
