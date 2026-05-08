import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import {
  Truck, MapPin, CheckCircle2, XCircle, Loader2, RefreshCw,
  User, Phone, Camera, FileText, ChevronRight,
} from 'lucide-react'
import {
  fetchDeliveries, fetchDrivers, assignDriver,
  updateDeliveryStatus, submitProofOfDelivery, trackDelivery,
} from '../../../services/platformApi'

const STATUS_COLORS = {
  ASSIGNED: '#3B82F6',
  PICKED_UP: '#8B5CF6',
  EN_ROUTE: '#F59E0B',
  DELIVERED: '#1A936F',
  FAILED: '#EF4444',
}

const STATUS_FLOW = ['ASSIGNED', 'PICKED_UP', 'EN_ROUTE', 'DELIVERED']

function nextStatus(current) {
  const idx = STATUS_FLOW.indexOf(current)
  return idx >= 0 && idx < STATUS_FLOW.length - 1 ? STATUS_FLOW[idx + 1] : null
}

export default function DeliveriesView() {
  const [deliveries, setDeliveries] = useState([])
  const [drivers, setDrivers] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [assigning, setAssigning] = useState(null)   // delivery id being assigned
  const [selectedDriver, setSelectedDriver] = useState({})
  const [trackingId, setTrackingId] = useState(null)
  const [trackingData, setTrackingData] = useState(null)
  const [proofModal, setProofModal] = useState(null)  // delivery id
  const [proofUrl, setProofUrl] = useState('')
  const [proofType, setProofType] = useState('PHOTO')

  const load = useCallback(async () => {
    try {
      const [dels, drvs] = await Promise.all([fetchDeliveries(), fetchDrivers()])
      setDeliveries(dels)
      setDrivers(drvs)
    } catch {
      toast.error('Failed to load deliveries')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const availableDrivers = drivers.filter(d => d.active && !d.on_delivery)

  const filtered = deliveries.filter(d =>
    filter === 'all' || d.status === filter
  )

  // Active (non-delivered) deliveries
  const activeCount = deliveries.filter(d => d.status !== 'DELIVERED' && d.status !== 'FAILED').length

  async function handleAssign(delivery) {
    const driverId = selectedDriver[delivery.id]
    if (!driverId) { toast.error('Select a driver first'); return }
    setAssigning(delivery.id)
    try {
      await assignDriver(delivery.order_id, driverId)
      toast.success(`Driver assigned to #${delivery.order_ref || delivery.order_id}`)
      load()
    } catch (err) {
      toast.error(err.message)
    } finally {
      setAssigning(null)
    }
  }

  async function handleStatusAdvance(delivery) {
    const next = nextStatus(delivery.status)
    if (!next) return
    try {
      await updateDeliveryStatus(delivery.id, next)
      toast.success(`Delivery → ${next}`)
      load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleMarkFailed(delivery) {
    try {
      await updateDeliveryStatus(delivery.id, 'FAILED')
      toast.error(`Delivery marked as FAILED`)
      load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleTrack(deliveryId) {
    setTrackingId(deliveryId)
    try {
      const data = await trackDelivery(deliveryId)
      setTrackingData(data)
    } catch {
      toast.error('Failed to load tracking data')
      setTrackingId(null)
    }
  }

  async function handleSubmitProof() {
    if (!proofUrl.trim()) { toast.error('Proof URL required'); return }
    try {
      await submitProofOfDelivery(proofModal, proofUrl, proofType)
      toast.success('Proof of delivery submitted')
      setProofModal(null)
      setProofUrl('')
      load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading deliveries...</div>

  return (
    <div style={{ padding: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>
            {deliveries.length} deliveries · {activeCount} active · {availableDrivers.length} drivers free
          </span>
          <button className="icon-btn" onClick={load}><RefreshCw size={15} /></button>
        </div>
      </div>

      {/* Filters */}
      <div className="orders-filters" style={{ marginBottom: 16 }}>
        {['all', 'ASSIGNED', 'PICKED_UP', 'EN_ROUTE', 'DELIVERED', 'FAILED'].map(s => (
          <button
            key={s}
            className={`btn btn-sm ${filter === s ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setFilter(s)}
          >
            {s === 'all' ? 'All' : s.replace('_', ' ')}
            {s !== 'all' && deliveries.filter(d => d.status === s).length > 0 && (
              <span style={{ marginLeft: 4, opacity: 0.7, fontSize: 11 }}>
                ({deliveries.filter(d => d.status === s).length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tracking Modal */}
      {trackingId && trackingData && (
        <div style={{
          background: 'var(--bg-overlay)', borderRadius: 12, padding: 20,
          marginBottom: 16, border: '1px solid var(--border)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ margin: 0 }}>Tracking Delivery #{trackingData.delivery_id}</h3>
            <button className="btn btn-ghost btn-sm" onClick={() => { setTrackingId(null); setTrackingData(null) }}>Close</button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
            <div className="stat-card">
              <div className="stat-label">Status</div>
              <div style={{ color: STATUS_COLORS[trackingData.status] || '#888', fontWeight: 700 }}>
                {trackingData.status?.replace('_', ' ')}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Driver</div>
              <div>{trackingData.driver?.name || 'Unassigned'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Order</div>
              <div>{trackingData.order?.order_ref || '—'}</div>
            </div>
          </div>
          {trackingData.driver && (
            <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>
              <Truck size={13} style={{ display: 'inline', marginRight: 4 }} />
              {trackingData.driver.vehicle_type} {trackingData.driver.vehicle_no}
              {trackingData.driver.phone && (
                <span style={{ marginLeft: 12 }}><Phone size={13} style={{ display: 'inline', marginRight: 4 }} />{trackingData.driver.phone}</span>
              )}
            </div>
          )}
          {trackingData.last_location && (
            <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>
              <MapPin size={13} style={{ display: 'inline', marginRight: 4 }} />
              Last: {trackingData.last_location.lat}, {trackingData.last_location.lng}
              <span style={{ marginLeft: 8 }}>
                ({new Date(trackingData.last_location.recorded_at).toLocaleTimeString('en-IN')})
              </span>
            </div>
          )}
          {trackingData.location_history?.length > 1 && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {trackingData.location_history.length} GPS pings recorded
            </div>
          )}
        </div>
      )}

      {/* Proof of Delivery Modal */}
      {proofModal && (
        <div style={{
          background: 'var(--bg-overlay)', borderRadius: 12, padding: 20,
          marginBottom: 16, border: '1px solid var(--border)',
        }}>
          <h3 style={{ marginBottom: 12 }}>Proof of Delivery — #{proofModal}</h3>
          <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Proof URL (photo/signature)</label>
              <input
                className="form-input"
                value={proofUrl}
                onChange={e => setProofUrl(e.target.value)}
                placeholder="https://storage.example.com/proof/delivery-123.jpg"
              />
            </div>
            <div className="form-group" style={{ width: 140 }}>
              <label className="form-label">Type</label>
              <select className="form-input" value={proofType} onChange={e => setProofType(e.target.value)}>
                <option value="PHOTO">Photo</option>
                <option value="SIGNATURE">Signature</option>
              </select>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary btn-sm" onClick={handleSubmitProof}>
              <Camera size={14} /> Submit Proof
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => { setProofModal(null); setProofUrl('') }}>Cancel</button>
          </div>
        </div>
      )}

      {/* Deliveries List */}
      {filtered.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 0', fontSize: 14 }}>
          No deliveries found
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {filtered.map(d => {
            const color = STATUS_COLORS[d.status] || '#888'
            const canAdvance = STATUS_FLOW.includes(d.status) && nextStatus(d.status)
            return (
              <div key={d.id} className="menu-item-row" style={{ padding: '12px 16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1 }}>
                  {/* Status indicator */}
                  <div style={{
                    width: 10, height: 10, borderRadius: '50%', background: color,
                    flexShrink: 0, boxShadow: `0 0 8px ${color}44`,
                  }} />

                  {/* Delivery info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 600, fontFamily: 'monospace', fontSize: 13 }}>
                        #{d.order_ref || d.order_id}
                      </span>
                      <span style={{
                        fontSize: 11, background: `${color}22`, color,
                        padding: '2px 8px', borderRadius: 6, fontWeight: 600,
                      }}>
                        {d.status?.replace('_', ' ')}
                      </span>
                      {d.order_type && (
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {d.order_type}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                      {d.address_json?.address || d.customer_phone || 'No address'}
                      {d.order_total_cents > 0 && (
                        <span style={{ marginLeft: 12 }}>₹{d.order_total_cents / 100}</span>
                      )}
                    </div>
                  </div>

                  {/* Driver info / assignment */}
                  <div style={{ width: 180, flexShrink: 0 }}>
                    {d.driver_name ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                        <User size={13} />
                        <span style={{ fontWeight: 600 }}>{d.driver_name}</span>
                      </div>
                    ) : d.status === 'ASSIGNED' ? (
                      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Awaiting driver</span>
                    ) : (
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <select
                          className="form-input"
                          style={{ fontSize: 12, padding: '4px 8px', flex: 1 }}
                          value={selectedDriver[d.id] || ''}
                          onChange={e => setSelectedDriver(prev => ({ ...prev, [d.id]: e.target.value }))}
                        >
                          <option value="">Assign driver...</option>
                          {availableDrivers.map(drv => (
                            <option key={drv.id} value={drv.id}>
                              {drv.name} ({drv.vehicle_type || 'N/A'})
                            </option>
                          ))}
                        </select>
                        <button
                          className="btn btn-sm btn-primary"
                          style={{ fontSize: 11, padding: '4px 10px' }}
                          disabled={assigning === d.id || !selectedDriver[d.id]}
                          onClick={() => handleAssign(d)}
                        >
                          {assigning === d.id ? <Loader2 size={12} className="spin" /> : 'Assign'}
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                    {canAdvance && (
                      <button
                        className="btn btn-sm btn-primary"
                        style={{ fontSize: 11, padding: '4px 10px' }}
                        onClick={() => handleStatusAdvance(d)}
                      >
                        <ChevronRight size={13} /> {nextStatus(d.status)?.replace('_', ' ')}
                      </button>
                    )}
                    {d.status !== 'DELIVERED' && d.status !== 'FAILED' && d.driver_id && (
                      <button
                        className="btn btn-sm btn-ghost"
                        style={{ fontSize: 11, color: '#EF4444', padding: '4px 10px' }}
                        onClick={() => handleMarkFailed(d)}
                      >
                        <XCircle size={13} />
                      </button>
                    )}
                    <button
                      className="btn btn-sm btn-ghost"
                      style={{ fontSize: 11, padding: '4px 10px' }}
                      onClick={() => handleTrack(d.id)}
                    >
                      <MapPin size={13} />
                    </button>
                    {d.status === 'EN_ROUTE' && !d.proof_url && (
                      <button
                        className="btn btn-sm btn-ghost"
                        style={{ fontSize: 11, padding: '4px 10px' }}
                        onClick={() => setProofModal(d.id)}
                      >
                        <Camera size={13} />
                      </button>
                    )}
                    {d.proof_url && (
                      <a
                        href={d.proof_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--brand-green)' }}
                      >
                        <FileText size={13} /> Proof
                      </a>
                    )}
                    {d.status === 'DELIVERED' && (
                      <CheckCircle2 size={16} style={{ color: 'var(--brand-green)' }} />
                    )}
                  </div>
                </div>

                {/* Timestamps */}
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, paddingLeft: 22, display: 'flex', gap: 16 }}>
                  {d.assigned_at && <span>Assigned: {new Date(d.assigned_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}</span>}
                  {d.picked_up_at && <span>Picked up: {new Date(d.picked_up_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}</span>}
                  {d.delivered_at && <span>Delivered: {new Date(d.delivered_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}</span>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
