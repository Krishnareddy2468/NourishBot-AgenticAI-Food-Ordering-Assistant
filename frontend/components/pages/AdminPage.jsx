'use client'
/**
 * Admin portal: sidebar nav with real views.
 */

import { useState, useEffect, startTransition, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import {
  LayoutDashboard, ShoppingCart, ClipboardList, BarChart3,
  ChefHat, Tag, Bell, RefreshCw, Zap, UtensilsCrossed, LogOut, Loader2,
  Users, ArrowLeftRight, Truck, CalendarClock,
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import { useRouter } from 'next/navigation'
import { useWebSocket } from '../../hooks/useWebSocket'
import { fetchBackendOrders, updateBackendOrderState, mapUiStateToBackend } from '../../services/platformApi'
import DashboardView from './views/DashboardView'
import OrdersView from './views/OrdersView'
import MenuView from './views/MenuView'
import EmployeesView from './views/EmployeesView'
import DeliveriesView from './views/DeliveriesView'
import AnalyticsView from './views/AnalyticsView'

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'orders', label: 'Scheduled Orders', icon: ClipboardList },
  { id: 'kitchen', label: 'Kitchen Queue', icon: ChefHat },
  { id: 'menu', label: 'Menu', icon: Tag },
  { id: 'employees', label: 'Canteen Team', icon: Users },
  { id: 'deliveries', label: 'Home Deliveries', icon: Truck },
  { id: 'analytics', label: 'Fulfillment Analytics', icon: BarChart3 },
]

function LiveClock() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  return <div className="time-chip">{t.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
}

function KDSView({ orders, upsertOrder }) {
  const buckets = {
    Pending: orders.filter(o => o.status === 'Pending'),
    Preparing: orders.filter(o => o.status === 'Preparing' || o.status === 'Accepted'),
    Ready: orders.filter(o => o.status === 'Ready'),
  }
  const colors = { Pending: '#F59E0B', Preparing: '#3B82F6', Ready: '#1A936F' }

  async function move(order, next) {
    if (!order.backendState) { toast.error('Read-only row'); return }
    try {
      const updated = await updateBackendOrderState(order.id, mapUiStateToBackend(next))
      upsertOrder(updated)
      toast.success(`${order.orderRef || order.id} → ${next}`)
    } catch (err) { toast.error(err.message) }
  }

  return (
    <div className="kds-layout" style={{ padding: 16 }}>
      {Object.entries(buckets).map(([status, items]) => (
        <div key={status} className="kds-column">
          <div className="kds-col-header">
            <div className="kds-col-title">
              <div className="kds-col-indicator" style={{ background: colors[status] }} />{status}
            </div>
            <span className="badge" style={{ background: `${colors[status]}22`, color: colors[status] }}>{items.length}</span>
          </div>
          <div className="kds-items">
            {items.length === 0 && <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, padding: '24px 0' }}>No orders</div>}
            {items.map(o => (
              <div key={o.id} className="kds-card">
                <div className="kds-card-header">
                  <span className="kds-order-id">#{o.orderRef || o.id}</span>
                  <span className="kds-customer">{o.customer}</span>
                </div>
                <div className="kds-items-list">
                  {(o.items || []).map((it, idx) => (
                    <div key={idx} className="kds-item">{it.qty}x {it.name}</div>
                  ))}
                </div>
                <div className="kds-actions">
                  {status === 'Pending' && (
                    <>
                      <button className="btn btn-ghost btn-sm" style={{ flex: 1, color: '#EF4444', fontSize: 11 }} onClick={() => move(o, 'Cancelled')}>Reject</button>
                      <button className="btn btn-primary btn-sm" style={{ flex: 1, fontSize: 11 }} onClick={() => move(o, 'Preparing')}>Accept</button>
                    </>
                  )}
                  {status === 'Preparing' && (
                    <button className="btn btn-primary btn-sm" style={{ width: '100%', fontSize: 11 }} onClick={() => move(o, 'Ready')}>→ Ready</button>
                  )}
                  {status === 'Ready' && (
                    o.orderType === 'DELIVERY'
                      ? <button className="btn btn-ghost btn-sm" style={{ width: '100%', fontSize: 11 }} disabled>Awaiting driver dispatch</button>
                      : <button className="btn btn-primary btn-sm" style={{ width: '100%', fontSize: 11, background: 'var(--brand-green)' }} onClick={() => move(o, 'Delivered')}>✓ Collected</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default function AdminPage() {
  const { user, logout } = useAuth()
  const router = useRouter()
  const [activeView, setActiveView] = useState('dashboard')
  const [globalSearch] = useState('')
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const { on, connected } = useWebSocket()

  const loadOrders = useCallback(async (silent = false) => {
    if (silent) setRefreshing(true)
    try {
      const data = await fetchBackendOrders()
      startTransition(() => setOrders(data))
    } catch (err) {
      if (!silent) toast.error(err.message || 'Failed to load orders')
    } finally {
      setRefreshing(false)
      setLoading(false)
    }
  }, [])

  function upsertOrder(order) {
    setOrders(prev => [order, ...prev.filter(o => o.id !== order.id)])
  }

  useEffect(() => { loadOrders() }, [loadOrders])

  useEffect(() => {
    const handler = evt => {
      if (evt.event_type?.startsWith('order')) loadOrders(true)
    }
    return on('*', handler)
  }, [on, loadOrders])

  const pendingCount = orders.filter(o => o.status === 'Pending').length
  const scheduledCount = orders.filter(o => o.status === 'Scheduled').length
  const preparingCount = orders.filter(o => o.status === 'Preparing' || o.status === 'Accepted').length
  const readyCount = orders.filter(o => o.status === 'Ready').length
  const revenueToday = orders.reduce((sum, order) => sum + Number(order.price || 0), 0)

  function renderView() {
    switch (activeView) {
      case 'dashboard': return <DashboardView orders={orders} />
      case 'orders': return <OrdersView orders={orders} upsertOrder={upsertOrder} globalSearch={globalSearch} />
      case 'menu': return <MenuView />
      case 'kitchen': return <KDSView orders={orders} upsertOrder={upsertOrder} />
      case 'employees': return <EmployeesView />
      case 'deliveries': return <DeliveriesView />
      case 'analytics': return <AnalyticsView />
      default: return <DashboardView orders={orders} />
    }
  }

  return (
    <div className="admin-page">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <UtensilsCrossed size={18} />
          <span>NourishBot</span>
        </div>
        <nav className="sidebar-nav">
          {NAV.map((item) => {
            const NavIcon = item.icon
            return (
              <button
                key={item.id}
                className={`sidebar-item ${activeView === item.id ? 'active' : ''}`}
                onClick={() => setActiveView(item.id)}
              >
                <NavIcon size={15} />
                <span>{item.label}</span>
                {item.id === 'orders' && pendingCount > 0 && (
                  <span className="sidebar-badge">{pendingCount}</span>
                )}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-user">{user?.email}</div>
          <button className="sidebar-item" onClick={() => router.push('/workspace')} title="Switch role">
            <ArrowLeftRight size={15} /> Switch Role
          </button>
          <button className="sidebar-item" onClick={logout}>
            <LogOut size={15} /> Logout
          </button>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="topbar-title">
            <h2>{NAV.find(i => i.id === activeView)?.label || 'Dashboard'}</h2>
            <p>{new Date().toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' })}</p>
          </div>
          <div className="topbar-actions">
            <div className={`ops-chip ${connected ? 'online' : 'offline'}`}>
              <span className="ops-chip-dot" />
              {connected ? 'Live sync on' : 'Reconnecting'}
            </div>
            <div className="ops-chip warm">
              <CalendarClock size={13} />
              {scheduledCount} upcoming
            </div>
            <div className="ops-chip warm">
              <ShoppingCart size={13} />
              {pendingCount} queued
            </div>
            <div className="ops-chip cool">
              <ChefHat size={13} />
              {preparingCount} in kitchen
            </div>
            <div className="ops-chip green">
              <Truck size={13} />
              {readyCount} ready for fulfillment
            </div>
            {pendingCount > 0 && <div className="live-ticker"><Zap size={12} />{pendingCount} pending</div>}
            <button className={`icon-btn ${refreshing ? 'is-spinning' : ''}`} onClick={() => loadOrders(true)} title="Refresh">
              <RefreshCw size={15} />
            </button>
            <button className="icon-btn"><Bell size={15} /><div className="notif-dot" /></button>
            <LiveClock />
          </div>
        </header>
        <div className="content">
          {loading ? (
            <div className="page-loading"><Loader2 className="spin" size={24} />Loading...</div>
          ) : (
            <>
              <section className="ops-hero">
                <div>
                  <span className="eyebrow">Operations pulse</span>
                  <h3>Upcoming meals activate into one live preparation queue.</h3>
                  <p>Scheduled orders stay out of the kitchen until their activation window, then move through preparation and fulfillment.</p>
                </div>
                <div className="ops-hero-stats">
                  <article>
                    <strong>{orders.length}</strong>
                    <span>Total orders</span>
                  </article>
                  <article>
                    <strong>{scheduledCount}</strong>
                    <span>Upcoming meals</span>
                  </article>
                  <article>
                    <strong>₹{revenueToday.toLocaleString('en-IN')}</strong>
                    <span>Order value tracked</span>
                  </article>
                  <article>
                    <strong>{connected ? 'Live' : 'Retrying'}</strong>
                    <span>Realtime link</span>
                  </article>
                </div>
              </section>
              {renderView()}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
