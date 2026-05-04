/**
 * Waiter view — table map, sessions, add items, fire-to-kitchen, generate bill.
 */

import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Plus, Minus, Flame, Receipt, Users, Loader2, CheckCircle2, XCircle } from 'lucide-react'
import {
  fetchTables, fetchMenu, fetchActiveSessions,
  openTableSession, addTableSessionOrder, fireTableSession,
  generateTableInvoice, closeTableSession,
} from '../../services/platformApi'
import { useWebSocket } from '../../hooks/useWebSocket'

const TABLE_COLORS = { AVAILABLE: '#1A936F', OCCUPIED: '#F59E0B', RESERVED: '#8B5CF6', INACTIVE: '#6B7280' }

export default function WaiterPage() {
  const [tables, setTables] = useState([])
  const [sessions, setSessions] = useState([])
  const [menu, setMenu] = useState([])
  const [selectedTable, setSelectedTable] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [cart, setCart] = useState([])
  const [loading, setLoading] = useState(true)
  const { on } = useWebSocket()

  // Realtime: listen for session/order updates
  useEffect(() => {
    const handler = (evt) => {
      if (evt.event_type?.startsWith('table_session') || evt.event_type?.startsWith('order')) {
        loadData()
      }
    }
    on('*', handler)
    return () => {}
  }, [on])

  const loadData = useCallback(async () => {
    try {
      const [t, s, m] = await Promise.all([fetchTables(), fetchActiveSessions(), fetchMenu()])
      setTables(t)
      setSessions(s)
      setMenu(m)
    } catch (err) {
      toast.error('Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

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
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleGenerateBill() {
    if (!activeSession) return
    try {
      const invoice = await generateTableInvoice(activeSession.id)
      toast.success(`Bill generated — ₹${(invoice.total_cents / 100).toFixed(2)}`)
      await closeTableSession(activeSession.id)
      setActiveSession(null)
      setSelectedTable(null)
      loadData()
    } catch (err) {
      toast.error(err.message)
    }
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={32} />Loading...</div>

  return (
    <div className="waiter-page">
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

      {/* Menu + Cart (when session active) */}
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

            <button className="btn btn-secondary" style={{ marginTop: 16 }} onClick={handleGenerateBill}>
              <Receipt size={14} /> Generate Bill & Close
            </button>
          </section>
        </div>
      )}
    </div>
  )
}
