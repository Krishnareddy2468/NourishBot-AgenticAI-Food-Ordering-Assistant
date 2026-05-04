/**
 * Waiter view — table map, sessions, add items, fire-to-kitchen,
 * kitchen readiness tracking, billing with payment gate.
 */

import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Plus, Minus, Flame, Receipt, Users, Loader2, CheckCircle2, XCircle, ArrowLeftRight, LogOut, Clock, Bell } from 'lucide-react'
import {
  fetchTables, fetchMenu, fetchActiveSessions, fetchSessionOrders,
  openTableSession, addTableSessionOrder, fireTableSession,
  generateTableInvoice, closeTableSession,
} from '../../services/platformApi'
import { useWebSocket } from '../../hooks/useWebSocket'
import { useAuth } from '../../context/AuthContext'
import { useNavigate } from 'react-router-dom'

const TABLE_COLORS = { AVAILABLE: '#1A936F', OCCUPIED: '#F59E0B', RESERVED: '#8B5CF6', INACTIVE: '#6B7280' }

const ITEM_STATUS_CHIP = {
  PENDING:      { label: 'Pending',   color: '#F59E0B', bg: 'rgba(245,158,11,0.12)' },
  IN_PROGRESS:  { label: 'Cooking',   color: '#3B82F6', bg: 'rgba(59,130,246,0.12)' },
  DONE:         { label: 'Done',      color: '#1A936F', bg: 'rgba(26,147,111,0.12)' },
  CANCELLED:    { label: 'Cancelled',  color: '#EF4444', bg: 'rgba(239,68,68,0.12)' },
}

export default function WaiterPage() {
  const { logout } = useAuth()
  const navigate = useNavigate()
  const [tables, setTables] = useState([])
  const [sessions, setSessions] = useState([])
  const [menu, setMenu] = useState([])
  const [selectedTable, setSelectedTable] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [sessionOrders, setSessionOrders] = useState([])
  const [cart, setCart] = useState([])
  const [loading, setLoading] = useState(true)
  const { on } = useWebSocket()

  // Realtime: listen for session/order/item updates
  useEffect(() => {
    const handler = (evt) => {
      if (evt.event_type?.startsWith('table_session') || evt.event_type?.startsWith('order')) {
        loadData()
      }
    }
    on('*', handler)
    return () => {}
  }, [on])

  const loadData = useCallback(async (silent = false) => {
    try {
      const [t, s, m] = await Promise.all([fetchTables(), fetchActiveSessions(), fetchMenu()])
      setTables(t)
      setSessions(s)
      setMenu(m)
    } catch (err) {
      if (!silent) toast.error('Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  // Reload session orders when session changes or data refreshes
  const loadSessionOrders = useCallback(async () => {
    if (!activeSession?.id) { setSessionOrders([]); return }
    try {
      const orders = await fetchSessionOrders(activeSession.id)
      setSessionOrders(orders)
    } catch {
      setSessionOrders([])
    }
  }, [activeSession])

  useEffect(() => { loadData() }, [loadData])
  useEffect(() => { loadSessionOrders() }, [loadSessionOrders])

  // Count ready-to-serve orders for the bell indicator
  const readyOrders = sessionOrders.filter(o => o.all_items_done && o.status !== 'DELIVERED')
  const hasReady = readyOrders.length > 0

  async function handleOpenSession(table) {
    const guests = prompt('Number of guests:', '2')
    if (!guests) return
    try {
      const session = await openTableSession(table.id, parseInt(guests))
      setActiveSession(session)
      setSelectedTable(table)
      setCart([])
      toast.success(`Session opened — Table ${table.table_number || table.name}`)
      loadData()
    } catch (err) {
      toast.error(err.message)
    }
  }

  function addToCart(item) {
    setCart(prev => {
      const existing = prev.find(c => c.id === item.id)
      if (existing) return prev.map(c => c.id === item.id ? { ...c, qty: c.qty + 1 } : c)
      return [...prev, { ...item, qty: 1 }]
    })
  }

  function removeFromCart(itemId) {
    setCart(prev => {
      const existing = prev.find(c => c.id === itemId)
      if (existing && existing.qty > 1) return prev.map(c => c.id === itemId ? { ...c, qty: c.qty - 1 } : c)
      return prev.filter(c => c.id !== itemId)
    })
  }

  async function handleSendToKitchen() {
    if (!activeSession || cart.length === 0) return
    try {
      const items = cart.map(c => ({ item_id: c.id, qty: c.qty }))
      await addTableSessionOrder(activeSession.id, items)
      await fireTableSession(activeSession.id)
      setCart([])
      toast.success('Order fired to kitchen!')
      loadSessionOrders()
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleGenerateBill() {
    if (!activeSession) return
    // Check if any orders are still cooking
    const cooking = sessionOrders.some(o =>
      o.items?.some(i => i.status === 'PENDING' || i.status === 'IN_PROGRESS')
    )
    if (cooking) {
      toast.error('Some items are still being prepared. Wait for all items to be done before billing.')
      return
    }
    try {
      const invoice = await generateTableInvoice(activeSession.id)
      toast.success(`Bill generated — ₹${(invoice.total_cents / 100).toFixed(2)}`)
      setActiveSession(null)
      setSelectedTable(null)
      setSessionOrders([])
      loadData()
    } catch (err) {
      toast.error(err.message)
    }
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={32} />Loading...</div>

  return (
    <div className="waiter-page">
      {/* Top bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Users size={18} /> Waiter Portal
          {hasReady && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              background: 'rgba(26,147,111,0.15)', color: '#1A936F',
              padding: '2px 10px', borderRadius: 20, fontSize: 12, fontWeight: 700,
              animation: 'pulse 2s infinite',
            }}>
              <Bell size={12} /> {readyOrders.length} Ready to Serve
            </span>
          )}
        </h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>
            <ArrowLeftRight size={14} /> Switch Role
          </button>
          <button className="btn btn-ghost btn-sm" onClick={logout}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </div>

      {/* Table Map */}
      <section className="waiter-section">
        <h3>Table Map</h3>
        <div className="table-grid">
          {tables.map(t => {
            const session = sessions.find(s => s.table_id === t.id && s.status === 'OPEN')
            const color = TABLE_COLORS[t.status] || TABLE_COLORS.AVAILABLE
            return (
              <button
                key={t.id}
                className={`table-tile ${selectedTable?.id === t.id ? 'selected' : ''}`}
                style={{ borderColor: color }}
                onClick={() => {
                  if (session) {
                    setActiveSession(session)
                    setSelectedTable(t)
                    setCart([])
                  } else if (t.status === 'AVAILABLE') {
                    handleOpenSession(t)
                  } else {
                    toast('Table not available')
                  }
                }}
              >
                <div className="table-tile-name">{t.table_number || t.name}</div>
                <div className="table-tile-status" style={{ color }}>{t.status}</div>
                {session && <div className="table-tile-guests"><Users size={12} />{session.guests}</div>}
              </button>
            )
          })}
        </div>
      </section>

      {/* Menu + Cart + Kitchen Status (when session active) */}
      {activeSession && (
        <div className="waiter-order-area">
          <section className="waiter-section">
            <h3>Menu — Add Items</h3>
            <div className="waiter-menu-grid">
              {menu.map(item => (
                <button key={item.id} className="menu-quick-add" onClick={() => addToCart(item)}>
                  <span>{item.name}</span>
                  <span className="menu-quick-price">₹{item.price_cents / 100}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="waiter-section">
            <h3>Current Ticket</h3>
            {cart.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No items yet. Tap menu items to add.</p>
            ) : (
              <div className="waiter-cart">
                {cart.map(c => (
                  <div key={c.id} className="waiter-cart-item">
                    <span>{c.name}</span>
                    <div className="waiter-cart-qty">
                      <button className="btn btn-ghost btn-sm" onClick={() => removeFromCart(c.id)}><Minus size={14} /></button>
                      <span>{c.qty}</span>
                      <button className="btn btn-ghost btn-sm" onClick={() => addToCart(c)}><Plus size={14} /></button>
                    </div>
                    <span>₹{(c.price_cents / 100) * c.qty}</span>
                  </div>
                ))}
                <div className="waiter-cart-total">
                  Total: ₹{cart.reduce((sum, c) => sum + (c.price_cents / 100) * c.qty, 0).toFixed(2)}
                </div>
                <div className="waiter-cart-actions">
                  <button className="btn btn-primary" onClick={handleSendToKitchen}>
                    <Flame size={14} /> Fire to Kitchen
                  </button>
                </div>
              </div>
            )}
          </section>

          {/* ── Kitchen Readiness Panel ──────────────────────────────────── */}
          <section className="waiter-section">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              Kitchen Status
              {hasReady && (
                <span style={{
                  background: '#1A936F', color: 'white', fontSize: 11,
                  padding: '1px 8px', borderRadius: 10, fontWeight: 700,
                }}>
                  {readyOrders.length} Ready
                </span>
              )}
            </h3>
            {sessionOrders.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No orders sent to kitchen yet.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {sessionOrders.map(order => {
                  const allDone = order.all_items_done
                  return (
                    <div key={order.id} style={{
                      background: allDone ? 'rgba(26,147,111,0.08)' : 'var(--bg-overlay)',
                      border: `1px solid ${allDone ? 'rgba(26,147,111,0.3)' : 'var(--border)'}`,
                      borderRadius: 10, padding: 12,
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <span style={{ fontWeight: 700, fontSize: 13 }}>#{order.order_ref}</span>
                        {allDone ? (
                          <span style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                            color: '#1A936F', fontSize: 12, fontWeight: 700,
                          }}>
                            <CheckCircle2 size={13} /> Ready to Serve
                          </span>
                        ) : (
                          <span style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                            color: '#3B82F6', fontSize: 12,
                          }}>
                            <Clock size={13} /> Preparing
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                        {(order.items || []).map(item => {
                          const chip = ITEM_STATUS_CHIP[item.status] || ITEM_STATUS_CHIP.PENDING
                          return (
                            <div key={item.id} style={{
                              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                              fontSize: 12, padding: '2px 0',
                            }}>
                              <span>{item.qty}x {item.name}</span>
                              <span style={{
                                fontSize: 10, fontWeight: 600,
                                color: chip.color, background: chip.bg,
                                padding: '1px 7px', borderRadius: 8,
                              }}>
                                {chip.label}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          <button className="btn btn-secondary" style={{ marginTop: 16 }} onClick={handleGenerateBill}>
            <Receipt size={14} /> Generate Bill & Close
          </button>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
      `}</style>
    </div>
  )
}
