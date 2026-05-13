'use client'
import { useState } from 'react'
import { toast } from 'react-hot-toast'
import { Search } from 'lucide-react'
import { updateBackendOrderState, mapUiStateToBackend } from '../../../services/platformApi'

const STATUS_OPTIONS = ['Pending', 'Accepted', 'Preparing', 'Ready', 'Delivered', 'Cancelled']
const STATUS_COLORS = {
  Pending: '#F59E0B', Accepted: '#3B82F6', Preparing: '#8B5CF6',
  Ready: '#1A936F', Delivered: '#22C55E', Cancelled: '#EF4444',
}

export default function OrdersView({ orders, upsertOrder, globalSearch }) {
  const [filter, setFilter] = useState('all')

  const filtered = orders.filter(o => {
    const matchSearch = !globalSearch ||
      o.customer?.toLowerCase().includes(globalSearch.toLowerCase()) ||
      String(o.id).includes(globalSearch) ||
      (o.orderRef || '').toLowerCase().includes(globalSearch.toLowerCase())
    const matchFilter = filter === 'all' || o.status === filter
    return matchSearch && matchFilter
  })

  async function handleStatusChange(order, next) {
    if (!order.backendState) {
      toast.error('Excel history rows are read-only')
      return
    }
    try {
      const updated = await updateBackendOrderState(order.id, mapUiStateToBackend(next))
      upsertOrder(updated)
      toast.success(`${order.orderRef || order.id} → ${next}`)
    } catch (err) {
      toast.error(err.message || 'Failed to update')
    }
  }

  return (
    <div className="orders-view">
      <div className="orders-filters">
        {['all', ...STATUS_OPTIONS].map(s => (
          <button
            key={s}
            className={`btn btn-sm ${filter === s ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setFilter(s)}
          >
            {s === 'all' ? 'All' : s}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: 12 }}>
          {filtered.length} orders
        </span>
      </div>

      <div className="orders-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Ref</th>
              <th>Customer</th>
              <th>Items</th>
              <th>Total</th>
              <th>Type</th>
              <th>Time</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>No orders</td></tr>
            )}
            {filtered.map(o => (
              <tr key={o.id}>
                <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--brand-primary)' }}>
                  #{o.orderRef || o.id}
                </td>
                <td>
                  <div style={{ fontWeight: 600 }}>{o.customer}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{o.phone}</div>
                </td>
                <td style={{ fontSize: 12, maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {o.item}
                </td>
                <td style={{ fontWeight: 600 }}>₹{Number(o.price || 0).toLocaleString('en-IN')}</td>
                <td>
                  <span style={{ fontSize: 11, background: 'var(--bg-overlay)', padding: '2px 8px', borderRadius: 8 }}>
                    {o.orderType || 'DELIVERY'}
                  </span>
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{o.dateTime}</td>
                <td>
                  <select
                    className="order-status-select"
                    value={o.status}
                    style={{ borderColor: STATUS_COLORS[o.status] || 'var(--border)' }}
                    onChange={e => handleStatusChange(o, e.target.value)}
                  >
                    {STATUS_OPTIONS.map(s => <option key={s}>{s}</option>)}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
