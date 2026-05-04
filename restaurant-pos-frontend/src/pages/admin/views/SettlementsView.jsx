import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Loader2, RefreshCw, TrendingUp, IndianRupee } from 'lucide-react'
import { fetchBackendOrders } from '../../../services/platformApi'

export default function SettlementsView() {
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const data = await fetchBackendOrders()
      setOrders(data)
    } catch (err) {
      toast.error('Failed to load settlement data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const delivered = orders.filter(o => o.status === 'Delivered')
  const totalRevenue = delivered.reduce((s, o) => s + (o.price || 0), 0)
  const avgOrderValue = delivered.length > 0 ? totalRevenue / delivered.length : 0
  const cancelled = orders.filter(o => o.status === 'Cancelled')

  // Group by order type
  const byType = delivered.reduce((acc, o) => {
    const t = o.orderType || 'OTHER'
    acc[t] = (acc[t] || { count: 0, revenue: 0 })
    acc[t].count++
    acc[t].revenue += o.price || 0
    return acc
  }, {})

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading settlements...</div>

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>Settlement Summary</h3>
        <button className="icon-btn" onClick={load}><RefreshCw size={15} /></button>
      </div>

      {/* KPI cards */}
      <div className="stat-grid" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-label">Delivered Orders</div>
          <div className="stat-value" style={{ color: 'var(--brand-green)' }}>{delivered.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Revenue</div>
          <div className="stat-value" style={{ color: 'var(--brand-primary)' }}>₹{totalRevenue.toLocaleString('en-IN')}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Order Value</div>
          <div className="stat-value">₹{avgOrderValue.toFixed(0)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Cancelled</div>
          <div className="stat-value" style={{ color: '#EF4444' }}>{cancelled.length}</div>
        </div>
      </div>

      {/* Revenue by type */}
      {Object.keys(byType).length > 0 && (
        <div className="chart-card" style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>Revenue by Order Type</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {Object.entries(byType).map(([type, data]) => {
              const pct = totalRevenue > 0 ? (data.revenue / totalRevenue) * 100 : 0
              return (
                <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 90, fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>{type}</div>
                  <div style={{ flex: 1, background: 'var(--bg-overlay)', borderRadius: 6, height: 10, overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: 'var(--grad-brand)', borderRadius: 6, transition: 'width 0.4s ease' }} />
                  </div>
                  <div style={{ minWidth: 80, textAlign: 'right', fontSize: 12, fontWeight: 700 }}>₹{data.revenue.toLocaleString('en-IN')}</div>
                  <div style={{ minWidth: 40, textAlign: 'right', fontSize: 11, color: 'var(--text-muted)' }}>{pct.toFixed(1)}%</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Delivered orders table */}
      <div className="section-header">
        <div className="section-title">Delivered Orders</div>
      </div>
      {delivered.length === 0 && (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0', fontSize: 14 }}>
          No delivered orders yet.
        </div>
      )}
      {delivered.length > 0 && (
        <div className="orders-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Order Ref</th>
                <th>Customer</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {delivered.map(o => (
                <tr key={o.id}>
                  <td style={{ fontFamily: 'monospace', color: 'var(--brand-primary)', fontWeight: 700 }}>
                    #{o.orderRef || o.id}
                  </td>
                  <td>
                    <div style={{ fontWeight: 600 }}>{o.customer}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{o.phone}</div>
                  </td>
                  <td>
                    <span style={{ fontSize: 11, background: 'var(--bg-overlay)', padding: '2px 8px', borderRadius: 6 }}>
                      {o.orderType || 'DELIVERY'}
                    </span>
                  </td>
                  <td style={{ fontWeight: 700, color: 'var(--brand-green)' }}>₹{o.price?.toLocaleString('en-IN')}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{o.dateTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
