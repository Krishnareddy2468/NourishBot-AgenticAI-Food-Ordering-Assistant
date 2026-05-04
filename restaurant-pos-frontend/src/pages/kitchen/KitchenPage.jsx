/**
 * Kitchen KDS v2 — item-level status tracking with station views.
 * Real-time WebSocket updates for order/item status changes.
 */

import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { ChefHat, CheckCircle2, Clock, Loader2, AlertTriangle, Flame } from 'lucide-react'
import { fetchKitchenOrders, updateOrderItemStatus, mapUiStateToBackend, updateBackendOrderState } from '../../services/platformApi'
import { useWebSocket } from '../../hooks/useWebSocket'

const ITEM_STATUS = {
  PENDING: { label: 'Pending', color: '#F59E0B', icon: Clock },
  IN_PROGRESS: { label: 'Cooking', color: '#3B82F6', icon: Flame },
  DONE: { label: 'Done', color: '#1A936F', icon: CheckCircle2 },
  CANCELLED: { label: 'Cancelled', color: '#EF4444', icon: XCircle },
}

const STATION_FILTERS = ['All', 'Main Course', 'Starters', 'Beverages', 'Desserts']

export default function KitchenPage() {
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const [stationFilter, setStationFilter] = useState('All')
  const { on } = useWebSocket()

  const loadOrders = useCallback(async () => {
    try {
      const data = await fetchKitchenOrders()
      setOrders(data)
    } catch (err) {
      toast.error('Failed to load kitchen orders')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadOrders() }, [loadOrders])

  // Realtime updates
  useEffect(() => {
    const handler = (evt) => {
      if (evt.event_type?.startsWith('order') || evt.event_type?.startsWith('order_item')) {
        loadOrders()
      }
    }
    on('*', handler)
  }, [on, loadOrders])

  async function handleItemStatusChange(orderId, itemId, newStatus) {
    try {
      await updateOrderItemStatus(orderId, itemId, newStatus)
      toast.success(`Item → ${newStatus}`)
      loadOrders()
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleMarkOrderReady(orderId) {
    try {
      await updateBackendOrderState(orderId, 'READY')
      toast.success('Order marked Ready!')
      loadOrders()
    } catch (err) {
      toast.error(err.message)
    }
  }

  // Filter orders that have kitchen-relevant items
  const kitchenOrders = orders.filter(o =>
    o.status === 'Accepted' || o.status === 'Preparing'
  )

  // Get all items across orders with their statuses
  const allItems = kitchenOrders.flatMap(order =>
    (order.items || []).map(item => ({
      ...item,
      orderId: order.id,
      orderRef: order.orderRef || order.id,
      customer: order.customer,
      orderType: order.orderType,
      itemStatus: item.status || 'PENDING',
    }))
  )

  const filteredItems = stationFilter === 'All'
    ? allItems
    : allItems.filter(i => i.category === stationFilter || i.type === stationFilter)

  // Group by item status for station-style view
  const buckets = {
    PENDING: filteredItems.filter(i => i.itemStatus === 'PENDING'),
    IN_PROGRESS: filteredItems.filter(i => i.itemStatus === 'IN_PROGRESS'),
    DONE: filteredItems.filter(i => i.itemStatus === 'DONE'),
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={32} />Loading...</div>

  return (
    <div className="kitchen-page">
      <div className="kitchen-header">
        <h2><ChefHat size={20} /> Kitchen Display</h2>
        <div className="kitchen-station-tabs">
          {STATION_FILTERS.map(s => (
            <button
              key={s}
              className={`btn btn-sm ${stationFilter === s ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setStationFilter(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="kds-v2-columns">
        {Object.entries(buckets).map(([status, items]) => {
          const cfg = ITEM_STATUS[status] || ITEM_STATUS.PENDING
          const Icon = cfg.icon
          return (
            <div key={status} className="kds-v2-column">
              <div className="kds-v2-col-header" style={{ borderColor: cfg.color }}>
                <Icon size={16} style={{ color: cfg.color }} />
                <span>{cfg.label}</span>
                <span className="badge" style={{ background: `${cfg.color}22`, color: cfg.color }}>{items.length}</span>
              </div>
              <div className="kds-v2-items">
                {items.length === 0 && <div className="kds-empty">No items</div>}
                {items.map(item => (
                  <div key={`${item.orderId}-${item.id}`} className="kds-v2-card">
                    <div className="kds-v2-card-header">
                      <span className="kds-v2-order-ref">#{item.orderRef}</span>
                      <span className="kds-v2-customer">{item.customer}</span>
                    </div>
                    <div className="kds-v2-item-name">
                      {item.qty || 1}x {item.name || item.item_name_snapshot}
                    </div>
                    {item.modifiers_json && (
                      <div className="kds-v2-mods">
                        {JSON.stringify(item.modifiers_json)}
                      </div>
                    )}
                    <div className="kds-v2-actions">
                      {status === 'PENDING' && (
                        <button className="btn btn-sm btn-primary" onClick={() => handleItemStatusChange(item.orderId, item.id, 'IN_PROGRESS')}>
                          Start Cooking
                        </button>
                      )}
                      {status === 'IN_PROGRESS' && (
                        <button className="btn btn-sm" style={{ background: '#1A936F', color: 'white' }} onClick={() => handleItemStatusChange(item.orderId, item.id, 'DONE')}>
                          <CheckCircle2 size={14} /> Done
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>

      {/* Orders ready to be marked as READY */}
      {kitchenOrders.length > 0 && (
        <section className="kitchen-orders-ready">
          <h3>Orders — Mark Ready When All Items Done</h3>
          <div className="kitchen-order-list">
            {kitchenOrders.map(order => {
              const allDone = (order.items || []).every(i => i.status === 'DONE')
              return (
                <div key={order.id} className="kitchen-order-row">
                  <span>#{order.orderRef || order.id}</span>
                  <span>{order.customer}</span>
                  <span>{order.item}</span>
                  <button
                    className="btn btn-sm"
                    disabled={!allDone}
                    style={{ background: allDone ? '#1A936F' : '#6B7280', color: 'white' }}
                    onClick={() => handleMarkOrderReady(order.id)}
                  >
                    <CheckCircle2 size={14} /> Mark Ready
                  </button>
                </div>
              )
            })}
          </div>
        </section>
      )}
    </div>
  )
}

function XCircle(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>
    </svg>
  )
}
