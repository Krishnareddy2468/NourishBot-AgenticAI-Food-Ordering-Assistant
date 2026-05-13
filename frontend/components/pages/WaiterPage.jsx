'use client'
/**
 * Waiter view — table map, sessions, add items, fire-to-kitchen,
 * kitchen readiness tracking, billing with payment gate.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { toast } from 'react-hot-toast'
import {
  Plus, Minus, Flame, Users, Loader2, CheckCircle2, ArrowLeftRight, LogOut,
  Clock, Bell, RefreshCw, ReceiptText, ChefHat, Wifi, WifiOff, CloudOff, Upload,
} from 'lucide-react'
import {
  fetchTables, fetchMenu, fetchActiveSessions, fetchSessionOrders,
  openTableSession, addTableSessionOrder, fireTableSession, generateTableInvoice,
} from '../../services/platformApi'
import {
  buildOfflineOrderEntry,
  enqueueOfflineOrder,
  getOfflineOrderQueue,
  getWaiterCache,
  removeOfflineOrder,
  setWaiterCache,
  updateOfflineOrder,
} from '../../services/offlineOrderQueue'
import { useWebSocket } from '../../hooks/useWebSocket'
import { useAuth } from '../../context/AuthContext'
import { useRouter } from 'next/navigation'

const TABLE_COLORS = { AVAILABLE: '#1A936F', OCCUPIED: '#F59E0B', RESERVED: '#8B5CF6', INACTIVE: '#6B7280' }

const ITEM_STATUS_CHIP = {
  PENDING: { label: 'Pending', color: '#F59E0B', bg: 'rgba(245,158,11,0.12)' },
  IN_PROGRESS: { label: 'Cooking', color: '#3B82F6', bg: 'rgba(59,130,246,0.12)' },
  DONE: { label: 'Done', color: '#1A936F', bg: 'rgba(26,147,111,0.12)' },
  CANCELLED: { label: 'Cancelled', color: '#EF4444', bg: 'rgba(239,68,68,0.12)' },
}

export default function WaiterPage() {
  const { logout, user } = useAuth()
  const router = useRouter()
  const [tables, setTables] = useState([])
  const [sessions, setSessions] = useState([])
  const [menu, setMenu] = useState([])
  const [selectedTable, setSelectedTable] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [sessionOrders, setSessionOrders] = useState([])
  const [cart, setCart] = useState([])
  const [offlineQueue, setOfflineQueue] = useState(() => getOfflineOrderQueue())
  const [loading, setLoading] = useState(true)
  const [actionState, setActionState] = useState({ openTableId: null, firing: false, billing: false, refreshing: false, syncingOffline: false })
  const [isOfflineMode, setIsOfflineMode] = useState(() => typeof navigator !== 'undefined' ? !navigator.onLine : false)
  const { on, connected } = useWebSocket(user?.restaurant_id || 1)

  const loadData = useCallback(async (silent = false) => {
    if (silent) {
      setActionState(prev => ({ ...prev, refreshing: true }))
    }
    try {
      const [nextTables, nextSessions, nextMenu] = await Promise.all([
        fetchTables(),
        fetchActiveSessions(),
        fetchMenu(),
      ])
      setTables(nextTables)
      setSessions(nextSessions)
      setMenu(nextMenu)
      setWaiterCache({ tables: nextTables, sessions: nextSessions, menu: nextMenu })
      setIsOfflineMode(false)
    } catch (_err) {
      const cached = getWaiterCache()
      if (cached.tables.length || cached.sessions.length || cached.menu.length) {
        setTables(cached.tables)
        setSessions(cached.sessions)
        setMenu(cached.menu)
        setIsOfflineMode(true)
        if (!silent) toast('Offline mode: using last saved floor data')
      } else if (!silent) {
        toast.error(_err.message || 'Failed to load waiter data')
      }
    } finally {
      setActionState(prev => ({ ...prev, refreshing: false }))
      setLoading(false)
    }
  }, [])

  const loadSessionOrders = useCallback(async () => {
    if (!activeSession?.id) {
      setSessionOrders([])
      return
    }
    try {
      const orders = await fetchSessionOrders(activeSession.id)
      setSessionOrders(orders)
    } catch {
      setSessionOrders([])
    }
  }, [activeSession])

  useEffect(() => { loadData() }, [loadData])
  useEffect(() => { loadSessionOrders() }, [loadSessionOrders])

  useEffect(() => {
    const handleOnline = () => setIsOfflineMode(false)
    const handleOffline = () => setIsOfflineMode(true)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  useEffect(() => {
    const handler = (evt) => {
      if (evt.event_type?.startsWith('table_session') || evt.event_type?.startsWith('order')) {
        loadData(true)
      }
    }
    return on('*', handler)
  }, [on, loadData])

  const syncOfflineOrders = useCallback(async () => {
    const queue = getOfflineOrderQueue()
    if (!queue.length || actionState.syncingOffline) return

    setActionState(prev => ({ ...prev, syncingOffline: true }))
    let syncedCount = 0

    try {
      for (const entry of queue) {
        updateOfflineOrder(entry.queueId, { status: 'SYNCING' })
        setOfflineQueue(getOfflineOrderQueue())

        const items = entry.items.map(item => ({ item_id: item.id, qty: item.qty }))
        await addTableSessionOrder(entry.sessionId, items)
        await fireTableSession(entry.sessionId)

        removeOfflineOrder(entry.queueId)
        setOfflineQueue(getOfflineOrderQueue())
        syncedCount += 1
      }

      if (syncedCount > 0) {
        toast.success(`${syncedCount} offline order${syncedCount > 1 ? 's' : ''} synced`)
        await Promise.all([loadData(true), loadSessionOrders()])
      }
    } catch (err) {
      const syncingEntry = getOfflineOrderQueue().find(entry => entry.status === 'SYNCING')
      if (syncingEntry) {
        updateOfflineOrder(syncingEntry.queueId, { status: 'QUEUED' })
        setOfflineQueue(getOfflineOrderQueue())
      }

      if (!navigator.onLine) {
        setIsOfflineMode(true)
      } else {
        toast.error(err.message || 'Failed to sync offline orders')
      }
    } finally {
      setActionState(prev => ({ ...prev, syncingOffline: false }))
    }
  }, [actionState.syncingOffline, loadData, loadSessionOrders])

  useEffect(() => {
    if (!isOfflineMode && navigator.onLine && offlineQueue.length > 0) {
      syncOfflineOrders()
    }
  }, [isOfflineMode, offlineQueue.length, syncOfflineOrders])

  useEffect(() => {
    if (!activeSession?.id) return
    const liveSession = sessions.find(session => session.id === activeSession.id)
    if (!liveSession || liveSession.status !== 'OPEN') {
      setActiveSession(null)
      setSelectedTable(null)
      setSessionOrders([])
      setCart([])
      return
    }
    setActiveSession(liveSession)
  }, [sessions, activeSession])

  useEffect(() => {
    if (!selectedTable?.id) return
    const liveTable = tables.find(table => table.id === selectedTable.id)
    if (liveTable) setSelectedTable(liveTable)
  }, [tables, selectedTable])

  const readyOrders = sessionOrders.filter(order => order.all_items_done && order.status !== 'DELIVERED')
  const hasReady = readyOrders.length > 0
  const availableMenu = useMemo(() => menu.filter(item => item.available !== false), [menu])
  const cartTotal = cart.reduce((sum, item) => sum + (item.price_cents / 100) * item.qty, 0)
  const cartQty = cart.reduce((sum, item) => sum + item.qty, 0)
  const sessionOfflineQueue = offlineQueue.filter(entry => entry.sessionId === activeSession?.id)
  const inProgressItems = sessionOrders.reduce((sum, order) => (
    sum + (order.items || []).filter(item => item.status === 'PENDING' || item.status === 'IN_PROGRESS').length
  ), 0)
  const readyItems = sessionOrders.reduce((sum, order) => (
    sum + (order.items || []).filter(item => item.status === 'DONE').length
  ), 0)

  async function handleOpenSession(table) {
    const guests = prompt('Number of guests:', '2')
    if (!guests) return

    setActionState(prev => ({ ...prev, openTableId: table.id }))
    try {
      const session = await openTableSession(table.id, parseInt(guests, 10))
      setActiveSession(session)
      setSelectedTable(table)
      setCart([])
      toast.success(`Session opened for Table ${table.table_number || table.name}`)
      await loadData(true)
    } catch (err) {
      toast.error(err.message || 'Could not open table session')
    } finally {
      setActionState(prev => ({ ...prev, openTableId: null }))
    }
  }

  function addToCart(item) {
    setCart(prev => {
      const existing = prev.find(entry => entry.id === item.id)
      if (existing) {
        return prev.map(entry => entry.id === item.id ? { ...entry, qty: entry.qty + 1 } : entry)
      }
      return [...prev, { ...item, qty: 1 }]
    })
  }

  function removeFromCart(itemId) {
    setCart(prev => {
      const existing = prev.find(entry => entry.id === itemId)
      if (existing && existing.qty > 1) {
        return prev.map(entry => entry.id === itemId ? { ...entry, qty: entry.qty - 1 } : entry)
      }
      return prev.filter(entry => entry.id !== itemId)
    })
  }

  async function handleSendToKitchen() {
    if (!activeSession || cart.length === 0) return

    setActionState(prev => ({ ...prev, firing: true }))
    try {
      const items = cart.map(item => ({ item_id: item.id, qty: item.qty }))
      await addTableSessionOrder(activeSession.id, items)
      await fireTableSession(activeSession.id)
      setCart([])
      setIsOfflineMode(false)
      toast.success('Ticket fired to kitchen')
      await Promise.all([loadSessionOrders(), loadData(true)])
    } catch {
      const offlineEntry = buildOfflineOrderEntry({
        sessionId: activeSession.id,
        table: selectedTable,
        items: cart,
        guests: activeSession.guests,
      })
      const nextQueue = enqueueOfflineOrder(offlineEntry)
      setOfflineQueue(nextQueue)
      setCart([])
      setIsOfflineMode(true)
      toast('Network unavailable. Order saved offline and will sync automatically.')
    } finally {
      setActionState(prev => ({ ...prev, firing: false }))
    }
  }

  async function handleGenerateBill() {
    if (!activeSession) return

    const cooking = sessionOrders.some(order =>
      order.items?.some(item => item.status === 'PENDING' || item.status === 'IN_PROGRESS')
    )
    if (cooking) {
      toast.error('Wait until all kitchen items are done before generating the bill.')
      return
    }

    setActionState(prev => ({ ...prev, billing: true }))
    try {
      const invoice = await generateTableInvoice(activeSession.id)
      toast.success(`Bill generated — ₹${(invoice.total_cents / 100).toFixed(2)}`)
      setActiveSession(null)
      setSelectedTable(null)
      setSessionOrders([])
      setCart([])
      await loadData(true)
    } catch (err) {
      toast.error(err.message || 'Failed to generate bill')
    } finally {
      setActionState(prev => ({ ...prev, billing: false }))
    }
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={32} />Loading...</div>

  return (
    <div className="waiter-page">
      <div className="ops-shell">
        <div className="ops-shell-copy">
          <span className="eyebrow">Floor service</span>
          <h2><Users size={20} /> Waiter Portal</h2>
          <p>Run the dining floor from table opening to kitchen fire to a clean checkout.</p>
        </div>
        <div className="ops-shell-actions">
          <div className={`ops-chip ${connected ? 'online' : 'offline'}`}>
            {connected ? <Wifi size={13} /> : <WifiOff size={13} />}
            {connected ? 'Kitchen sync live' : 'Kitchen sync retrying'}
          </div>
          {offlineQueue.length > 0 && (
            <div className="ops-chip warm">
              <CloudOff size={13} /> {offlineQueue.length} offline queue
            </div>
          )}
          {hasReady && (
            <div className="ops-chip green">
              <Bell size={13} /> {readyOrders.length} ready to serve
            </div>
          )}
          <button className={`icon-btn ${actionState.refreshing ? 'is-spinning' : ''}`} onClick={() => loadData(true)} title="Refresh">
            <RefreshCw size={15} />
          </button>
          {offlineQueue.length > 0 && (
            <button className="btn btn-ghost btn-sm" onClick={syncOfflineOrders} disabled={actionState.syncingOffline}>
              {actionState.syncingOffline ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
              {actionState.syncingOffline ? 'Syncing...' : 'Sync Offline'}
            </button>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => router.push('/')}>
            <ArrowLeftRight size={14} /> Switch Role
          </button>
          <button className="btn btn-ghost btn-sm" onClick={logout}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </div>

      <section className="ops-hero waiter-hero">
        <div>
          <span className="eyebrow">Service flow</span>
          <h3>{activeSession ? `Table ${selectedTable?.table_number || selectedTable?.name} is active` : 'Pick a table to start a guest session'}</h3>
          <p>
            {activeSession
              ? `Guests: ${activeSession.guests}. Add items, fire the ticket, and watch kitchen progress before billing.`
              : 'Available tables can be opened instantly, and occupied ones keep their live order readiness here.'}
          </p>
          {isOfflineMode && (
            <div className="offline-banner">
              <CloudOff size={15} />
              Offline mode enabled. New restaurant orders are being queued locally for sync.
            </div>
          )}
        </div>
        <div className="ops-hero-stats">
          <article>
            <strong>{tables.filter(table => table.active !== false).length}</strong>
            <span>Floor tables</span>
          </article>
          <article>
            <strong>{sessions.length}</strong>
            <span>Open sessions</span>
          </article>
          <article>
            <strong>{offlineQueue.length || (hasReady ? readyOrders.length : inProgressItems)}</strong>
            <span>{offlineQueue.length ? 'Queued offline' : (hasReady ? 'Orders ready' : 'Items in kitchen')}</span>
          </article>
        </div>
      </section>

      <section className="waiter-section">
        <div className="waiter-section-head">
          <h3>Table Map</h3>
          <span>{tables.length} tables</span>
        </div>
        <div className="table-grid">
          {tables.map(table => {
            const session = sessions.find(entry => entry.table_id === table.id && entry.status === 'OPEN')
            const statusKey = session ? 'OCCUPIED' : table.status
            const color = TABLE_COLORS[statusKey] || TABLE_COLORS.AVAILABLE
            const isOpening = actionState.openTableId === table.id

            return (
              <button
                key={table.id}
                className={`table-tile ${selectedTable?.id === table.id ? 'selected' : ''}`}
                style={{ borderColor: color }}
                disabled={isOpening}
                onClick={() => {
                  if (session) {
                    setActiveSession(session)
                    setSelectedTable(table)
                    setCart([])
                  } else if (table.status === 'AVAILABLE') {
                    handleOpenSession(table)
                  } else {
                    toast('Table not available')
                  }
                }}
              >
                <div className="table-tile-name">{table.table_number || table.name}</div>
                <div className="table-tile-status" style={{ color }}>{statusKey}</div>
                <div className="table-tile-capacity">Seats {table.capacity || '—'}</div>
                {session && <div className="table-tile-guests"><Users size={12} />{session.guests}</div>}
                {isOpening && <div className="table-tile-loading"><Loader2 className="spin" size={14} />Opening</div>}
              </button>
            )
          })}
        </div>
      </section>

      {activeSession && (
        <div className="waiter-order-area">
          <section className="waiter-section">
            <div className="waiter-section-head">
              <h3>Menu</h3>
              <span>{availableMenu.length} items available</span>
            </div>
            <div className="waiter-menu-grid">
              {availableMenu.map(item => (
                <button key={item.id} className="menu-quick-add" onClick={() => addToCart(item)}>
                  <div>
                    <strong>{item.name}</strong>
                    <small>{item.category_name || item.category || 'Menu item'}</small>
                  </div>
                  <span className="menu-quick-price">₹{item.price_cents / 100}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="waiter-section">
            <div className="waiter-section-head">
              <h3>Current Ticket</h3>
              <span>{cartQty} items</span>
            </div>
            {cart.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No items yet. Tap menu items to add.</p>
            ) : (
              <div className="waiter-cart">
                {cart.map(item => (
                  <div key={item.id} className="waiter-cart-item">
                    <span>{item.name}</span>
                    <div className="waiter-cart-qty">
                      <button className="btn btn-ghost btn-sm" onClick={() => removeFromCart(item.id)}><Minus size={14} /></button>
                      <span>{item.qty}</span>
                      <button className="btn btn-ghost btn-sm" onClick={() => addToCart(item)}><Plus size={14} /></button>
                    </div>
                    <span>₹{((item.price_cents / 100) * item.qty).toFixed(2)}</span>
                  </div>
                ))}
                <div className="waiter-cart-total">Total: ₹{cartTotal.toFixed(2)}</div>
                <div className="waiter-cart-actions">
                  <button className="btn btn-primary" onClick={handleSendToKitchen} disabled={actionState.firing}>
                    {actionState.firing ? <Loader2 size={14} className="spin" /> : <Flame size={14} />}
                    {actionState.firing ? 'Sending...' : 'Fire to Kitchen'}
                  </button>
                  <button className="btn btn-ghost" onClick={() => setCart([])}>Clear Ticket</button>
                </div>
              </div>
            )}
          </section>

          <section className="waiter-section">
            <div className="waiter-section-head">
              <h3>Kitchen Status</h3>
              {hasReady && (
                <span className="waiter-ready-pill">
                  <Bell size={12} /> {readyOrders.length} Ready
                </span>
              )}
            </div>
            {sessionOrders.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No orders sent to kitchen yet.</p>
            ) : (
              <div className="waiter-status-stack">
                {sessionOrders.map(order => {
                  const allDone = order.all_items_done
                  return (
                    <div key={order.id} className={`waiter-order-status-card ${allDone ? 'done' : ''}`}>
                      <div className="waiter-order-status-head">
                        <span>#{order.order_ref}</span>
                        {allDone ? (
                          <span className="waiter-chip success"><CheckCircle2 size={13} /> Ready to serve</span>
                        ) : (
                          <span className="waiter-chip info"><Clock size={13} /> Preparing</span>
                        )}
                      </div>
                      <div className="waiter-status-list">
                        {(order.items || []).map(item => {
                          const chip = ITEM_STATUS_CHIP[item.status] || ITEM_STATUS_CHIP.PENDING
                          return (
                            <div key={item.id} className="waiter-status-row">
                              <span>{item.qty}x {item.name}</span>
                              <span style={{ color: chip.color, background: chip.bg }}>{chip.label}</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {sessionOfflineQueue.length > 0 && (
              <div className="offline-order-stack">
                <div className="waiter-section-head" style={{ marginTop: 16 }}>
                  <h3>Offline Queue</h3>
                  <span>{sessionOfflineQueue.length} pending sync</span>
                </div>
                {sessionOfflineQueue.map(entry => (
                  <div key={entry.queueId} className="offline-order-card">
                    <div className="offline-order-head">
                      <strong>{entry.tableName}</strong>
                      <span>{entry.status === 'SYNCING' ? 'Syncing' : 'Queued offline'}</span>
                    </div>
                    <div className="offline-order-lines">
                      {entry.items.map(item => (
                        <div key={`${entry.queueId}-${item.id}`} className="offline-order-line">
                          <span>{item.qty}x {item.name}</span>
                          <span>₹{(((item.price_cents || 0) / 100) * item.qty).toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="waiter-kpi-strip">
              <div>
                <ChefHat size={14} />
                <span>{inProgressItems} cooking</span>
              </div>
              <div>
                <CheckCircle2 size={14} />
                <span>{readyItems} ready items</span>
              </div>
              <div>
                <ReceiptText size={14} />
                <span>{sessionOrders.length} fired orders</span>
              </div>
            </div>

            <div className="waiter-billing-card">
              <div>
                <span className="eyebrow">Close the table</span>
                <h4>Generate bill only after the kitchen is clear</h4>
                <p>Billing stays blocked while any fired item is still pending or cooking.</p>
              </div>
              <button className="btn btn-primary" onClick={handleGenerateBill} disabled={actionState.billing}>
                {actionState.billing ? <Loader2 size={14} className="spin" /> : <ReceiptText size={14} />}
                {actionState.billing ? 'Generating...' : 'Generate Bill & Close'}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
