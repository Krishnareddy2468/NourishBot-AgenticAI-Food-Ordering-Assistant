/**
 * Admin portal — sidebar nav with real views.
 */

import { useState, useEffect, useMemo, startTransition } from 'react'
import { toast } from 'react-hot-toast'
import {
  LayoutDashboard, ShoppingCart, ClipboardList, BarChart3,
  ChefHat, Tag, Bell, RefreshCw, Zap, UtensilsCrossed, LogOut, Loader2,
  Table2, CalendarDays, Users, Receipt, Banknote, ArrowLeftRight, Truck,
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { useAuth } from '../../context/AuthContext'
import { useNavigate } from 'react-router-dom'
import { useWebSocket } from '../../hooks/useWebSocket'
import { fetchBackendOrders, updateBackendOrderState, mapUiStateToBackend } from '../../services/platformApi'
import DashboardView from './views/DashboardView'
import OrdersView from './views/OrdersView'
import MenuView from './views/MenuView'
import TablesView from './views/TablesView'
import ReservationsView from './views/ReservationsView'
import EmployeesView from './views/EmployeesView'
import InvoicesView from './views/InvoicesView'
import SettlementsView from './views/SettlementsView'
import DeliveriesView from './views/DeliveriesView'

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'orders', label: 'Orders', icon: ClipboardList },
  { id: 'deliveries', label: 'Deliveries', icon: Truck },
  { id: 'kitchen', label: 'KDS', icon: ChefHat },
  { id: 'tables', label: 'Tables', icon: Table2 },
  { id: 'reservations', label: 'Reservations', icon: CalendarDays },
  { id: 'menu', label: 'Menu', icon: Tag },
  { id: 'employees', label: 'Employees', icon: Users },
  { id: 'invoices', label: 'Invoices', icon: Receipt },
  { id: 'settlements', label: 'Settlements', icon: Banknote },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
]

function LiveClock() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  return <div className="time-chip">{t.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
}

function AnalyticsView({ orders }) {
  const byHour = useMemo(() => {
    const hours = Array.from({ length: 13 }, (_, i) => ({ hour: `${i + 10}h`, revenue: 0 }))
    orders.forEach(o => {
      const h = o.dateTime ? new Date(o.dateTime).getHours() - 10 : -1
      if (h >= 0 && h < 13) hours[h].revenue += o.price || 0
    })
    return hours
  }, [orders])
  return (
    <div style={{ padding: 16 }}>
      <h3 style={{ marginBottom: 16 }}>Revenue by Hour</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={byHour}>
          <defs>
            <linearGradient id="rev" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FF6B35" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#FF6B35" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="hour" tick={{ fontSize: 11, fill: '#888' }} />
          <YAxis tick={{ fontSize: 11, fill: '#888' }} />
          <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
          <Area type="monotone" dataKey="revenue" stroke="#FF6B35" fill="url(#rev)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
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
                    <button className="btn btn-primary btn-sm" style={{ width: '100%', fontSize: 11, background: 'var(--brand-green)' }} onClick={() => move(o, 'Delivered')}>✓ Delivered</button>
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
  const navigate = useNavigate()
  const [activeView, setActiveView] = useState('dashboard')
  const [globalSearch, setGlobalSearch] = useState('')
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const { on } = useWebSocket()

  async function loadOrders(silent = false) {
    try {
      const data = await fetchBackendOrders()
      startTransition(() => setOrders(data))
    } catch (err) {
      if (!silent) toast.error(err.message || 'Failed to load orders')
    } finally {
      setLoading(false)
    }
  }

  function upsertOrder(order) {
    setOrders(prev => [order, ...prev.filter(o => o.id !== order.id)])
  }

  useEffect(() => { loadOrders() }, [])

  useEffect(() => {
    const handler = evt => {
      if (evt.event_type?.startsWith('order')) loadOrders(true)
    }
    on('*', handler)
  }, [on])

  const pendingCount = orders.filter(o => o.status === 'Pending').length

  function renderView() {
    switch (activeView) {
      case 'dashboard': return <DashboardView orders={orders} />
      case 'orders': return <OrdersView orders={orders} upsertOrder={upsertOrder} globalSearch={globalSearch} />
      case 'deliveries': return <DeliveriesView />
      case 'menu': return <MenuView />
      case 'kitchen': return <KDSView orders={orders} upsertOrder={upsertOrder} />
      case 'tables': return <TablesView />
      case 'reservations': return <ReservationsView />
      case 'employees': return <EmployeesView />
      case 'invoices': return <InvoicesView />
      case 'settlements': return <SettlementsView />
      case 'analytics': return <AnalyticsView orders={orders} />
      default: return <DashboardView orders={orders} />
    }
  }

  return (
    <div className="admin-page">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <UtensilsCrossed size={18} />
          <span>Dzukku</span>
        </div>
        <nav className="sidebar-nav">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`sidebar-item ${activeView === id ? 'active' : ''}`}
              onClick={() => setActiveView(id)}
            >
              <Icon size={15} />
              <span>{label}</span>
              {id === 'orders' && pendingCount > 0 && (
                <span className="sidebar-badge">{pendingCount}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-user">{user?.email}</div>
          <button className="sidebar-item" onClick={() => navigate('/')} title="Switch role">
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
            {pendingCount > 0 && <div className="live-ticker"><Zap size={12} />{pendingCount} pending</div>}
            <button className="icon-btn" onClick={() => loadOrders()} title="Refresh"><RefreshCw size={15} /></button>
            <button className="icon-btn"><Bell size={15} /><div className="notif-dot" /></button>
            <LiveClock />
          </div>
        </header>
        <div className="content">
          {loading ? (
            <div className="page-loading"><Loader2 className="spin" size={24} />Loading...</div>
          ) : renderView()}
        </div>
      </div>
    </div>
  )
}
