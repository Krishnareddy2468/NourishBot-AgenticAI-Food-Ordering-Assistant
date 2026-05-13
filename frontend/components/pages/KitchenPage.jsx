'use client'
/**
 * Kitchen KDS v2 — item-level status tracking with station views.
 * Real-time WebSocket updates for order/item status changes.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { toast } from 'react-hot-toast'
import {
  ChefHat, CheckCircle2, Clock, Loader2, Flame, ArrowLeftRight, LogOut,
  XCircle, RefreshCw, Wifi, WifiOff, ScanSearch,
} from 'lucide-react'
import { fetchKitchenOrders, updateOrderItemStatus, updateBackendOrderState } from '../../services/platformApi'
import { useWebSocket } from '../../hooks/useWebSocket'
import { useAuth } from '../../context/AuthContext'
import { useRouter } from 'next/navigation'

const ITEM_STATUS = {
  PENDING: { label: 'Pending', color: '#F59E0B', icon: Clock },
  IN_PROGRESS: { label: 'Cooking', color: '#3B82F6', icon: Flame },
  DONE: { label: 'Done', color: '#1A936F', icon: CheckCircle2 },
  CANCELLED: { label: 'Cancelled', color: '#EF4444', icon: XCircle },
}

const STATION_FILTERS = ['All', 'DINE_IN', 'DELIVERY', 'PICKUP']

export default function KitchenPage() {
  const { logout, user } = useAuth()
  const router = useRouter()
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const [stationFilter, setStationFilter] = useState('All')
  const [actionKey, setActionKey] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const { on, connected } = useWebSocket(user?.restaurant_id || 1)

  const loadOrders = useCallback(async (silent = false) => {
    if (silent) setRefreshing(true)
    try {
      const data = await fetchKitchenOrders()
      setOrders(data)
    } catch (err) {
      toast.error(err.message || 'Failed to load kitchen orders', { id: 'kitchen-orders-load-error' })
    } finally {
      setRefreshing(false)
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadOrders() }, [loadOrders])

  useEffect(() => {
    const handler = (evt) => {
      if (evt.event_type?.startsWith('order')) {
        loadOrders(true)
      }
    }
    return on('*', handler)
  }, [on, loadOrders])

  async function handleItemStatusChange(orderId, itemId, newStatus) {
    const key = `${orderId}:${itemId}:${newStatus}`
    setActionKey(key)
    try {
      await updateOrderItemStatus(orderId, itemId, newStatus)
      toast.success(`Item → ${newStatus}`)
      await loadOrders(true)
    } catch (err) {
      toast.error(err.message || 'Failed to update item status')
    } finally {
      setActionKey('')
    }
  }

  async function handleMarkOrderReady(orderId) {
    const key = `ready:${orderId}`
    setActionKey(key)
    try {
      await updateBackendOrderState(orderId, 'READY')
      toast.success('Order marked ready')
      await loadOrders(true)
    } catch (err) {
      toast.error(err.message || 'Failed to mark order ready')
    } finally {
      setActionKey('')
    }
  }

  const kitchenOrders = useMemo(() => (
    orders.filter(order => order.status === 'ACCEPTED' || order.status === 'PREPARING' || order.status === 'CREATED')
  ), [orders])

  const allItems = useMemo(() => (
    kitchenOrders.flatMap(order =>
      (order.items || []).map(item => ({
        ...item,
        orderId: order.id,
        orderRef: order.orderRef || order.order_ref || order.id,
        customer: order.customer,
        orderType: order.orderType || order.order_type,
        itemStatus: item.status || 'PENDING',
      }))
    )
  ), [kitchenOrders])

  const filteredItems = stationFilter === 'All'
    ? allItems
    : allItems.filter(item => item.orderType === stationFilter)

  const buckets = {
    PENDING: filteredItems.filter(item => item.itemStatus === 'PENDING'),
    IN_PROGRESS: filteredItems.filter(item => item.itemStatus === 'IN_PROGRESS'),
    DONE: filteredItems.filter(item => item.itemStatus === 'DONE'),
  }

  const readyToExpo = kitchenOrders.filter(order => (order.items || []).length > 0 && (order.items || []).every(item => item.status === 'DONE'))

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={32} />Loading...</div>

  return (
    <div className="kitchen-page">
      <div className="ops-shell">
        <div className="ops-shell-copy">
          <span className="eyebrow">Back of house</span>
          <h2><ChefHat size={20} /> Kitchen Display</h2>
          <p>Track active tickets by item, move them through prep, and only release orders when every line item is done.</p>
        </div>
        <div className="ops-shell-actions">
          <div className={`ops-chip ${connected ? 'online' : 'offline'}`}>
            {connected ? <Wifi size={13} /> : <WifiOff size={13} />}
            {connected ? 'Realtime linked' : 'Realtime retrying'}
          </div>
          <button className={`icon-btn ${refreshing ? 'is-spinning' : ''}`} onClick={() => loadOrders(true)} title="Refresh">
            <RefreshCw size={15} />
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => router.push('/')}>
            <ArrowLeftRight size={14} /> Switch Role
          </button>
          <button className="btn btn-ghost btn-sm" onClick={logout}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </div>

      <section className="ops-hero kitchen-hero">
        <div>
          <span className="eyebrow">Kitchen pulse</span>
          <h3>{filteredItems.length} live line items on the board</h3>
          <p>Each action here updates waiter and admin views, so the pass stays aligned with the front of house.</p>
        </div>
        <div className="ops-hero-stats">
          <article>
            <strong>{buckets.PENDING.length}</strong>
            <span>Queued</span>
          </article>
          <article>
            <strong>{buckets.IN_PROGRESS.length}</strong>
            <span>Cooking</span>
          </article>
          <article>
            <strong>{readyToExpo.length}</strong>
            <span>Ready orders</span>
          </article>
        </div>
      </section>

      <div className="kitchen-header">
        <div className="kitchen-header-copy">
          <h3>Station filter</h3>
          <p>Filter the queue by working order type instead of empty menu categories.</p>
        </div>
        <div className="kitchen-station-tabs">
          {STATION_FILTERS.map(filter => (
            <button
              key={filter}
              className={`btn btn-sm ${stationFilter === filter ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setStationFilter(filter)}
            >
              <ScanSearch size={13} />
              {filter === 'All' ? 'All tickets' : filter.replace('_', ' ')}
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
                {items.map(item => {
                  const keyPrefix = `${item.orderId}:${item.id}`
                  return (
                    <div key={keyPrefix} className="kds-v2-card">
                      <div className="kds-v2-card-header">
                        <span className="kds-v2-order-ref">#{item.orderRef}</span>
                        <span className="kds-v2-customer">{item.customer}</span>
                      </div>
                      <div className="kds-v2-item-name">
                        {item.qty || 1}x {item.name || item.item_name_snapshot}
                      </div>
                      <div className="kds-v2-card-meta">
                        <span>{item.orderType?.replace('_', ' ') || 'Order'}</span>
                        <span>{item.itemStatus.replace('_', ' ')}</span>
                      </div>
                      {item.modifiers_json && (
                        <div className="kds-v2-mods">{JSON.stringify(item.modifiers_json)}</div>
                      )}
                      <div className="kds-v2-actions">
                        {status === 'PENDING' && (
                          <button
                            className="btn btn-sm btn-primary"
                            disabled={actionKey === `${keyPrefix}:IN_PROGRESS`}
                            onClick={() => handleItemStatusChange(item.orderId, item.id, 'IN_PROGRESS')}
                          >
                            {actionKey === `${keyPrefix}:IN_PROGRESS` ? <Loader2 size={14} className="spin" /> : 'Start cooking'}
                          </button>
                        )}
                        {status === 'IN_PROGRESS' && (
                          <button
                            className="btn btn-sm"
                            style={{ background: '#1A936F', color: 'white' }}
                            disabled={actionKey === `${keyPrefix}:DONE`}
                            onClick={() => handleItemStatusChange(item.orderId, item.id, 'DONE')}
                          >
                            {actionKey === `${keyPrefix}:DONE` ? <Loader2 size={14} className="spin" /> : <CheckCircle2 size={14} />}
                            {actionKey === `${keyPrefix}:DONE` ? 'Updating...' : 'Done'}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {kitchenOrders.length > 0 && (
        <section className="kitchen-orders-ready">
          <h3>Expedite Queue</h3>
          <div className="kitchen-order-list">
            {kitchenOrders.map(order => {
              const allDone = (order.items || []).every(item => item.status === 'DONE')
              const busy = actionKey === `ready:${order.id}`
              return (
                <div key={order.id} className={`kitchen-order-row ${allDone ? 'is-ready' : ''}`}>
                  <div>
                    <strong>#{order.orderRef || order.id}</strong>
                    <div className="kitchen-order-row-meta">{order.customer} • {(order.orderType || order.order_type || 'ORDER').replace('_', ' ')}</div>
                  </div>
                  <span className="kitchen-order-row-items">
                    {(order.items || []).map(item => `${item.qty}x ${item.name || item.item_name_snapshot}`).join(', ')}
                  </span>
                  <button
                    className="btn btn-sm"
                    disabled={!allDone || busy}
                    style={{ background: allDone ? '#1A936F' : '#6B7280', color: 'white' }}
                    onClick={() => handleMarkOrderReady(order.id)}
                  >
                    {busy ? <Loader2 size={14} className="spin" /> : <CheckCircle2 size={14} />}
                    {busy ? 'Updating...' : 'Mark Ready'}
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
