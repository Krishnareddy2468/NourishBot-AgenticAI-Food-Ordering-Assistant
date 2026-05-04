import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Calendar, Users, Loader2, RefreshCw } from 'lucide-react'
import { fetchReservations, updateReservationStatus } from '../../../services/platformApi'

const STATUS_COLORS = {
  CREATED: { bg: 'rgba(245,158,11,0.15)', color: '#F59E0B', border: 'rgba(245,158,11,0.3)' },
  CONFIRMED: { bg: 'rgba(26,147,111,0.15)', color: '#1A936F', border: 'rgba(26,147,111,0.3)' },
  CANCELLED: { bg: 'rgba(239,68,68,0.15)', color: '#EF4444', border: 'rgba(239,68,68,0.3)' },
  NO_SHOW: { bg: 'rgba(107,114,128,0.15)', color: '#6B7280', border: 'rgba(107,114,128,0.3)' },
}

const FILTERS = ['ALL', 'CREATED', 'CONFIRMED', 'CANCELLED', 'NO_SHOW']

export default function ReservationsView() {
  const [reservations, setReservations] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('ALL')
  const [updating, setUpdating] = useState(null)

  const load = useCallback(async () => {
    try {
      const data = await fetchReservations()
      setReservations(data)
    } catch (err) {
      toast.error('Failed to load reservations')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function handleStatusChange(id, status) {
    setUpdating(id)
    try {
      await updateReservationStatus(id, status)
      setReservations(prev => prev.map(r => r.id === id ? { ...r, status } : r))
      toast.success(`Reservation → ${status}`)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setUpdating(null)
    }
  }

  const filtered = filter === 'ALL' ? reservations : reservations.filter(r => r.status === filter)

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading reservations...</div>

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        {FILTERS.map(f => (
          <button
            key={f}
            className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: 12 }}>{filtered.length} reservations</span>
        <button className="icon-btn" onClick={load} title="Refresh"><RefreshCw size={15} /></button>
      </div>

      {filtered.length === 0 && (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 0', fontSize: 14 }}>
          No reservations found.
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {filtered.map(r => {
          const sc = STATUS_COLORS[r.status] || STATUS_COLORS.CREATED
          return (
            <div
              key={r.id}
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 12,
                padding: '14px 18px',
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                flexWrap: 'wrap',
              }}
            >
              {/* Ref */}
              <div style={{ fontFamily: 'monospace', color: 'var(--brand-primary)', fontWeight: 700, minWidth: 100 }}>
                {r.reservation_ref || `#${r.id}`}
              </div>

              {/* Customer */}
              <div style={{ flex: 1, minWidth: 140 }}>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{r.customer_name || 'Guest'}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{r.customer_phone || '—'}</div>
              </div>

              {/* Date/Time */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-secondary)', fontSize: 13 }}>
                <Calendar size={14} />
                <span>{r.date || '—'}</span>
                <span>{r.time ? r.time.slice(0, 5) : ''}</span>
              </div>

              {/* Guests */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)', fontSize: 13 }}>
                <Users size={14} />{r.guests}
              </div>

              {/* Special request */}
              {r.special_request && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  "{r.special_request}"
                </div>
              )}

              {/* Status badge */}
              <span style={{
                background: sc.bg, color: sc.color, border: `1px solid ${sc.border}`,
                borderRadius: 20, padding: '3px 10px', fontSize: 11, fontWeight: 700,
              }}>
                {r.status}
              </span>

              {/* Actions */}
              {r.status === 'CREATED' && (
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ color: '#1A936F', borderColor: 'rgba(26,147,111,0.4)' }}
                    disabled={updating === r.id}
                    onClick={() => handleStatusChange(r.id, 'CONFIRMED')}
                  >
                    Confirm
                  </button>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ color: '#EF4444', borderColor: 'rgba(239,68,68,0.4)' }}
                    disabled={updating === r.id}
                    onClick={() => handleStatusChange(r.id, 'CANCELLED')}
                  >
                    Cancel
                  </button>
                </div>
              )}
              {r.status === 'CONFIRMED' && (
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ color: '#6B7280' }}
                    disabled={updating === r.id}
                    onClick={() => handleStatusChange(r.id, 'NO_SHOW')}
                  >
                    No Show
                  </button>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ color: '#EF4444', borderColor: 'rgba(239,68,68,0.4)' }}
                    disabled={updating === r.id}
                    onClick={() => handleStatusChange(r.id, 'CANCELLED')}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
