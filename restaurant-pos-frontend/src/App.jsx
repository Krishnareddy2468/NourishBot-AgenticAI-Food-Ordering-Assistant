import { startTransition, useEffect, useMemo, useState } from 'react'
import { Toaster, toast } from 'react-hot-toast'
import {
  LayoutDashboard, ShoppingCart, UtensilsCrossed, ClipboardList,
  BarChart3, Grid3X3, ChefHat, Tag, Bell, Search, RefreshCw,
  TrendingUp, TrendingDown, Clock, CheckCircle2, Trash2, Plus,
  Minus, FileText, Zap, Star, Users, CalendarCheck, Receipt,
  AlertTriangle, Loader2, FileSpreadsheet, ArrowUpRight
} from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { useExcelData } from './hooks/useExcelData'
import {
  createBackendOrder,
  createBackendPaymentIntent,
  fetchBackendOrders,
  fetchBackendSettlement,
  mapUiStateToBackend,
  markBackendOrderPaid,
  updateBackendOrderState,
} from './services/platformApi'
import './index.css'

const STATUS_OPTIONS = ['Pending', 'Accepted', 'Preparing', 'Ready', 'Delivered', 'Cancelled']

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatCurrency(v) {
  return `Rs. ${Number(v || 0).toLocaleString('en-IN')}`
}

function LiveClock() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  return <div className="time-chip">{t.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
}

function KDSColumn({ title, items, color, renderActions }) {
  return (
    <div className="kds-column">
      <div className="kds-col-header">
        <div className="kds-col-title">
          <div className="kds-col-indicator" style={{ background: color }} />{title}
        </div>
        <span className="badge" style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}>{items.length}</span>
      </div>
      <div className="kds-items">
        {items.length === 0
          ? <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, padding: '40px 0' }}>No orders</div>
          : items.map(order => (
            <div key={order.id} className="kds-ticket anim-fade">
              <div className="kds-ticket-header">
                <span className="kds-ticket-id">{order.id}</span>
                <span className="kds-ticket-time" style={{ color }}><Clock size={11} style={{ display: 'inline', marginRight: 2 }} />{order.eta}</span>
              </div>
              <div className="kds-ticket-customer">{order.customer}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>{order.platform}</div>
              <div className="kds-ticket-items">
                {(order.items?.length ? order.items : [{ emoji: '🍽️', name: order.item, qty: order.qty }]).map((it, i) => (
                  <div key={i} className="kds-ticket-item">
                    <span>{it.emoji || '•'}</span><span>{it.name} ×{it.qty}</span>
                  </div>
                ))}
              </div>
              <div className="kds-ticket-actions">
                {renderActions ? renderActions(order) : null}
              </div>
            </div>
          ))}
      </div>
    </div>
  )
}

// ─── Loading Screen ────────────────────────────────────────────────────────────
function LoadingScreen() {
  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 20,
      background: 'var(--bg-base)',
    }}>
      <div style={{ fontSize: 56 }}>🍽️</div>
      <div style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", fontSize: 28, fontWeight: 800, background: 'var(--grad-brand)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        Dzukku POS
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-muted)', fontSize: 14 }}>
        <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
        Loading data from Excel…
      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}

// ─── Error Screen ──────────────────────────────────────────────────────────────
function ErrorScreen({ error, onRetry }) {
  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 16,
      background: 'var(--bg-base)',
    }}>
      <AlertTriangle size={48} color="var(--status-cancelled)" />
      <div style={{ fontSize: 20, fontWeight: 700 }}>Failed to load Excel data</div>
      <div style={{ color: 'var(--text-muted)', fontSize: 13, maxWidth: 400, textAlign: 'center' }}>{error}</div>
      <button className="btn btn-primary" onClick={onRetry}><RefreshCw size={14} /> Retry</button>
    </div>
  )
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'dashboard',   label: 'Dashboard',       icon: LayoutDashboard, section: 'OVERVIEW' },
  { id: 'pos',         label: 'POS & Orders',     icon: ShoppingCart,    section: 'OPERATIONS' },
  { id: 'kds',         label: 'Kitchen Display',  icon: ChefHat,         section: 'OPERATIONS' },
  { id: 'tables',      label: 'Table Map',        icon: Grid3X3,         section: 'OPERATIONS' },
  { id: 'reservations',label: 'Reservations',     icon: CalendarCheck,   section: 'OPERATIONS' },
  { id: 'orders',      label: 'Order History',    icon: ClipboardList,   section: 'RECORDS' },
  { id: 'invoices',    label: 'Invoices',         icon: Receipt,         section: 'RECORDS' },
  { id: 'menu',        label: 'Menu Manager',     icon: UtensilsCrossed, section: 'CATALOG' },
  { id: 'specials',    label: 'Special Offers',   icon: Tag,             section: 'CATALOG' },
  { id: 'employees',   label: 'Staff',            icon: Users,           section: 'ADMIN' },
  { id: 'analytics',   label: 'Analytics',        icon: BarChart3,       section: 'INSIGHTS' },
]

function Sidebar({ active, setActive, pendingCount, lastLoaded }) {
  const sections = [...new Set(NAV_ITEMS.map(i => i.section))]
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <div className="logo-icon">🍽️</div>
          <div className="logo-text">
            <span className="logo-name">Dzukku</span>
            <span className="logo-tagline">Restaurant POS</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {sections.map(section => {
          const items = NAV_ITEMS.filter(i => i.section === section)
          return (
            <div key={section}>
              <p className="sidebar-section-label">{section}</p>
              {items.map(item => {
                const Icon = item.icon
                return (
                  <div key={item.id} className={`nav-item ${active === item.id ? 'active' : ''}`}
                    onClick={() => setActive(item.id)}>
                    <Icon className="nav-icon" size={17} />
                    <span>{item.label}</span>
                    {item.id === 'kds' && pendingCount > 0 && (
                      <span className="nav-badge">{pendingCount}</span>
                    )}
                  </div>
                )
              })}
            </div>
          )
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="status-indicator">
          <FileSpreadsheet size={13} />
          <span style={{ fontSize: 11 }}>
            Excel synced {lastLoaded ? lastLoaded.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '—'}
          </span>
        </div>
      </div>
    </aside>
  )
}

// ─── Topbar ───────────────────────────────────────────────────────────────────
function Topbar({ activeView, pendingCount, globalSearch, setGlobalSearch, onRefresh }) {
  const label = NAV_ITEMS.find(i => i.id === activeView)?.label || 'Dashboard'
  return (
    <>
      <header className="topbar">
        <div className="topbar-title">
          <h2>{label}</h2>
          <p>{new Date().toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' })}</p>
        </div>
        <div className="search-bar">
          <Search size={14} color="var(--text-muted)" />
          <input value={globalSearch} onChange={e => setGlobalSearch(e.target.value)} placeholder="Search orders, customers…" />
        </div>
        <div className="topbar-actions">
          {pendingCount > 0 && (
            <div className="live-ticker"><Zap size={12} />{pendingCount} pending</div>
          )}
          <button className="icon-btn" onClick={onRefresh} title="Reload Excel"><RefreshCw size={15} /></button>
          <button className="icon-btn"><Bell size={15} /><div className="notif-dot" /></button>
          <LiveClock />
        </div>
      </header>
    </>
  )
}

// ─── Dashboard View ────────────────────────────────────────────────────────────
function DashboardView({ analytics, menu, orders, specials }) {
  const TODAY = analytics[analytics.length - 1] || {}
  const YESTERDAY = analytics[analytics.length - 2] || {}
  const revChange = YESTERDAY.revenue ? (((TODAY.revenue - YESTERDAY.revenue) / YESTERDAY.revenue) * 100).toFixed(1) : 0
  const ordChange = YESTERDAY.orders ? (((TODAY.orders - YESTERDAY.orders) / YESTERDAY.orders) * 100).toFixed(1) : 0

  const COLORS = ['#FF6B35', '#1A936F', '#7B2FBE', '#004E89', '#F59E0B']

  const categoryData = menu.reduce((acc, item) => {
    const ex = acc.find(a => a.name === item.category)
    if (ex) ex.value++; else acc.push({ name: item.category, value: 1 })
    return acc
  }, [])

  const deliveredCount = orders.filter(o => o.status === 'Delivered').length
  const pendingCount = orders.filter(o => o.status === 'Pending' || o.status === 'Preparing').length

  const metrics = [
    { label: "Today's Revenue",    value: formatCurrency(TODAY.revenue),  icon: '💰', cls: 'orange', change: `${revChange}%`, up: revChange > 0 },
    { label: "Today's Orders",     value: TODAY.orders || 0,              icon: '📋', cls: 'blue',   change: `${ordChange}%`, up: ordChange > 0 },
    { label: 'Avg Order Value',    value: formatCurrency(TODAY.avg),      icon: '📊', cls: 'green',  change: '+2.3%', up: true },
    { label: 'Online Payments',    value: TODAY.online || 0,              icon: '💳', cls: 'purple', change: null },
    { label: 'Menu Items',         value: menu.length,                    icon: '🍽️', cls: 'gold',   change: null },
    { label: 'Delivered Today',    value: deliveredCount,                 icon: '✅', cls: 'green',  change: null },
    { label: 'Specials Active',    value: specials.length,                icon: '🔥', cls: 'orange', change: null },
    { label: 'Pending Orders',     value: pendingCount,                   icon: '⏳', cls: 'gold',   change: null },
  ]

  // Top selling items from Sales Dashboard data
  const topItems = useMemo(() => {
    const map = {}
    orders.forEach(o => {
      const key = o.item || 'Unknown'
      map[key] = (map[key] || 0) + (o.qty || 1)
    })
    return Object.entries(map).sort((a, b) => b[1] - a[1]).slice(0, 5)
  }, [orders])

  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))' }}>
        {metrics.map((m, i) => (
          <div className="metric-card" key={i}>
            <div className={`metric-icon ${m.cls}`}>{m.icon}</div>
            <div className="metric-value">{m.value}</div>
            <div className="metric-label">{m.label}</div>
            {m.change !== null && (
              <div className={`metric-change ${m.up ? 'up' : 'down'}`}>
                {m.up ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                {m.change} vs yesterday
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="analytics-grid">
        {/* Revenue Trend Chart */}
        <div className="chart-card">
          <div className="chart-header">
            <div><div className="section-title">Revenue Trend</div><div className="section-sub">From Order Analytical sheet</div></div>
            <span className="badge badge-special">Live Excel</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={analytics}>
              <defs>
                <linearGradient id="gr1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF6B35" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#FF6B35" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => `Rs. ${(v / 1000).toFixed(0)}K`} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF', fontSize: 12 }} formatter={v => [formatCurrency(v), 'Revenue']} />
              <Area type="monotone" dataKey="revenue" stroke="#FF6B35" strokeWidth={2.5} fill="url(#gr1)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Orders Breakdown */}
        <div className="chart-card">
          <div className="chart-header">
            <div><div className="section-title">Online vs Cash Orders</div><div className="section-sub">Daily breakdown</div></div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={analytics} barSize={14}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF', fontSize: 12 }} />
              <Legend />
              <Bar dataKey="online" name="Online" fill="#FF6B35" radius={[4,4,0,0]} />
              <Bar dataKey="cash" name="Cash" fill="#1A936F" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Menu Category Pie */}
        <div className="chart-card">
          <div className="chart-header"><div className="section-title">Menu by Category</div></div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', alignItems: 'center' }}>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={categoryData} cx="50%" cy="50%" innerRadius={48} outerRadius={75} dataKey="value" paddingAngle={4}>
                  {categoryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF', fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {categoryData.map((cat, i) => (
                <div key={cat.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: COLORS[i % COLORS.length], flexShrink: 0 }} />
                  <span style={{ flex: 1, fontSize: 12, color: 'var(--text-secondary)' }}>{cat.name}</span>
                  <span style={{ fontSize: 12, fontWeight: 700 }}>{cat.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Top Items from Orders */}
        <div className="chart-card">
          <div className="chart-header"><div className="section-title">Top Ordered Items</div><div className="section-sub">From Orders sheet</div></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {topItems.length === 0
              ? <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No order data yet</div>
              : topItems.map(([name, qty], i) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 22, height: 22, borderRadius: 6, background: 'var(--bg-overlay)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 800, color: 'var(--brand-primary)', flexShrink: 0 }}>
                    {i + 1}
                  </div>
                  <div style={{ flex: 1, fontSize: 12, color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</div>
                  <span style={{ fontSize: 12, fontWeight: 700 }}>{qty} orders</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── POS View ─────────────────────────────────────────────────────────────────
function POSView({ menu, tables, upsertOrder }) {
  const cats = ['All', ...new Set(menu.map(m => m.category))]
  const [category, setCategory] = useState('All')
  const [menuSearch, setMenuSearch] = useState('')
  const [cart, setCart] = useState([])
  const [customer, setCustomer] = useState({ name: '', phone: '', type: 'Dine-In', table: tables[0]?.id || 'T01', platform: 'Telegram', address: '' })
  const [showInvoice, setShowInvoice] = useState(false)
  const [lastOrder, setLastOrder] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const filteredMenu = useMemo(() =>
    menu.filter(item => {
      const catOk = category === 'All' || item.category === category
      const searchOk = !menuSearch || item.name.toLowerCase().includes(menuSearch.toLowerCase())
      return catOk && searchOk
    }),
    [menu, category, menuSearch]
  )

  const addToCart = (item) => {
    setCart(prev => {
      const ex = prev.find(c => c.id === item.id)
      if (ex) return prev.map(c => c.id === item.id ? { ...c, qty: c.qty + 1 } : c)
      return [...prev, { ...item, qty: 1, finalPrice: item.isSpecial && item.specialPrice ? item.specialPrice : item.price }]
    })
    toast.success(`${item.name} added`, { duration: 1000, icon: item.emoji })
  }

  const updateQty = (id, delta) => {
    setCart(prev => prev.map(c => c.id === id ? { ...c, qty: Math.max(0, c.qty + delta) } : c).filter(c => c.qty > 0))
  }

  const cartSubtotal = cart.reduce((s, c) => s + c.finalPrice * c.qty, 0)
  const gst = Math.round(cartSubtotal * 0.05)
  const grandTotal = cartSubtotal + gst

  const placeOrder = async () => {
    if (!customer.name || !customer.phone) { toast.error('Fill customer name & phone'); return }
    if (!cart.length) { toast.error('Cart is empty'); return }

    setSubmitting(true)
    try {
      const platform = customer.type === 'Dine-In'
        ? `Table ${customer.table}`
        : customer.type === 'Delivery'
          ? customer.platform
          : 'Takeaway'

      const created = await createBackendOrder({
        tenant_id: 'RES_001',
        customer_id: customer.phone,
        customer_name: customer.name,
        phone: customer.phone,
        cart_items: cart.map(item => ({
          id: item.id,
          name: item.name,
          qty: item.qty,
          unit_price: item.finalPrice,
        })),
        fulfillment_type: customer.type === 'Dine-In' ? 'dine_in' : customer.type.toLowerCase(),
        special_instructions: '',
        address: customer.type === 'Delivery' ? customer.address : '',
        table_no: customer.type === 'Dine-In' ? customer.table : '',
        platform,
      })

      let paymentSession = null
      let settlement = created.settlement
      let persistedOrder = created
      try {
        paymentSession = await createBackendPaymentIntent({
          order_id: created.id,
          amount: created.price,
          currency: 'INR',
          payment_method_hint: 'upi',
        })
        if (customer.type !== 'Delivery') {
          persistedOrder = await markBackendOrderPaid(created.id)
        }
        settlement = await fetchBackendSettlement(created.id)
      } catch (paymentError) {
        toast.error(paymentError.message || 'Payment setup failed')
      }

      const orderForUi = { ...persistedOrder, paymentSession, settlement }
      upsertOrder(orderForUi)
      setLastOrder(orderForUi)
      setShowInvoice(true)
      setCart([])
      setCustomer(p => ({ ...p, name: '', phone: '', address: '' }))
      toast.success(`Order ${created.id} placed!`, { icon: '🎉' })
    } catch (error) {
      toast.error(error.message || 'Failed to place order')
    } finally {
      setSubmitting(false)
    }
  }

  const handleMarkPaid = async () => {
    if (!lastOrder?.id) return
    try {
      const paidOrder = await markBackendOrderPaid(lastOrder.id)
      const settlement = await fetchBackendSettlement(lastOrder.id)
      const nextOrder = { ...paidOrder, paymentSession: lastOrder.paymentSession, settlement }
      upsertOrder(nextOrder)
      setLastOrder(nextOrder)
      toast.success(`${lastOrder.id} marked paid`)
    } catch (error) {
      toast.error(error.message || 'Failed to confirm payment')
    }
  }

  return (
    <div className="anim-fade pos-layout">
      {/* Menu Panel */}
      <div className="menu-panel">
        <div className="menu-panel-header">
          <div className="search-bar" style={{ flex: '0 0 240px', margin: 0 }}>
            <Search size={13} color="var(--text-muted)" />
            <input value={menuSearch} onChange={e => setMenuSearch(e.target.value)} placeholder="Search menu…" />
          </div>
          <div className="category-pills" style={{ flex: 1, overflowX: 'auto' }}>
            {cats.map(cat => (
              <button key={cat} className={`cat-pill ${category === cat ? 'active' : ''}`} onClick={() => setCategory(cat)}>{cat}</button>
            ))}
          </div>
        </div>

        <div className="menu-grid">
          {filteredMenu.length === 0 ? (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
              No items found
            </div>
          ) : filteredMenu.map(item => (
            <div key={item.id} className="menu-item-card" onClick={() => addToCart(item)}>
              {item.isSpecial && <div className="special-ribbon">Special</div>}
              {item.stock <= 3 && item.stock > 0 && <div className="low-stock-badge">Low Stock</div>}
              {item.stock === 0 && <div className="low-stock-badge" style={{ background: 'rgba(239,68,68,0.35)', color: '#EF4444' }}>Out of Stock</div>}
              <span className="menu-item-emoji">{item.emoji}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                <div className={`veg-dot ${item.category === 'Non-Veg' ? 'nonveg' : 'veg'}`} />
                <div className="menu-item-name">{item.name}</div>
              </div>
              <div className="menu-item-desc">{item.description}</div>
              <div className="menu-item-footer">
                <div className="menu-item-price">
                  {formatCurrency(item.isSpecial && item.specialPrice ? item.specialPrice : item.price)}
                  {item.isSpecial && item.specialPrice && <span>{formatCurrency(item.price)}</span>}
                </div>
                <button className="add-btn" onClick={e => { e.stopPropagation(); addToCart(item) }}>+</button>
              </div>
              <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-muted)' }}>
                <Clock size={9} style={{ display: 'inline', marginRight: 2 }} />{item.prepTime} · Stock: {item.stock}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Cart Panel */}
      <div className="cart-panel">
        <div className="cart-header">
          <span className="cart-title">🛒 Current Order</span>
          <span className="cart-count">{cart.length} items</span>
        </div>

        <div className="cart-customer-section">
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Customer Name</label>
              <input className="form-input" placeholder="Full name" value={customer.name} onChange={e => setCustomer(p => ({ ...p, name: e.target.value }))} />
            </div>
            <div className="form-group">
              <label className="form-label">Phone</label>
              <input className="form-input" placeholder="10-digit" value={customer.phone} onChange={e => setCustomer(p => ({ ...p, phone: e.target.value }))} />
            </div>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Order Type</label>
              <select className="form-input" value={customer.type} onChange={e => setCustomer(p => ({ ...p, type: e.target.value }))}>
                <option>Dine-In</option><option>Takeaway</option><option>Delivery</option>
              </select>
            </div>
            {customer.type === 'Dine-In' ? (
              <div className="form-group">
                <label className="form-label">Table</label>
                <select className="form-input" value={customer.table} onChange={e => setCustomer(p => ({ ...p, table: e.target.value }))}>
                  {tables.map(t => <option key={t.id} value={t.id}>{t.id} ({t.capacity} pax)</option>)}
                </select>
              </div>
            ) : customer.type === 'Delivery' ? (
              <div className="form-group">
                <label className="form-label">Platform</label>
                <select className="form-input" value={customer.platform} onChange={e => setCustomer(p => ({ ...p, platform: e.target.value }))}>
                  <option>Telegram</option><option>WhatsApp</option>
                </select>
              </div>
            ) : null}
          </div>
        </div>

        <div className="cart-items">
          {cart.length === 0
            ? <div className="cart-empty"><div className="cart-empty-icon">🛒</div><span>Tap menu items to add</span></div>
            : cart.map(item => (
              <div key={item.id} className="cart-item-row">
                <div style={{ fontSize: 20 }}>{item.emoji}</div>
                <div className="cart-item-info">
                  <div className="cart-item-name">{item.name}</div>
                  <div className="cart-item-price">{formatCurrency(item.finalPrice)} each</div>
                </div>
                <div className="qty-control">
                  <button className="qty-btn" onClick={() => updateQty(item.id, -1)}><Minus size={10} /></button>
                  <span className="qty-num">{item.qty}</span>
                  <button className="qty-btn" onClick={() => updateQty(item.id, 1)}><Plus size={10} /></button>
                </div>
                <button className="remove-btn" onClick={() => setCart(p => p.filter(c => c.id !== item.id))}><Trash2 size={13} /></button>
              </div>
            ))}
        </div>

        <div className="cart-footer">
          <div className="cart-totals">
            <div className="cart-total-row"><span>Subtotal</span><span>{formatCurrency(cartSubtotal)}</span></div>
            <div className="cart-total-row"><span>GST (5%)</span><span>{formatCurrency(gst)}</span></div>
            <div className="cart-total-row grand"><span>Grand Total</span><span style={{ color: 'var(--brand-primary)' }}>{formatCurrency(grandTotal)}</span></div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => setCart([])}>Clear</button>
            <button className="btn btn-primary" style={{ flex: 2 }} onClick={placeOrder} disabled={submitting}>
              {submitting ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <CheckCircle2 size={14} />}
              {submitting ? 'Creating…' : 'Place Order'}
            </button>
          </div>
        </div>
      </div>

      {/* Invoice Modal */}
      {showInvoice && lastOrder && (
        <div className="modal-overlay" onClick={() => setShowInvoice(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="invoice-header">
              <div className="invoice-logo">Dzukku</div>
              <div className="invoice-sub">Restaurant · Bangalore</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                {new Date().toLocaleString('en-IN')}
              </div>
            </div>
            <hr className="invoice-divider" />
            <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Order ID</span><strong>{lastOrder.id}</strong></div>
            <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Customer</span><span>{lastOrder.customer}</span></div>
            <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Phone</span><span>{lastOrder.phone}</span></div>
            <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Order Type</span><span>{lastOrder.platform}</span></div>
            <hr className="invoice-divider" />
            {lastOrder.items?.map((item, i) => (
              <div key={i} className="invoice-row">
                <span>{item.emoji} {item.name} ×{item.qty}</span>
                <span>{formatCurrency(item.finalPrice * item.qty)}</span>
              </div>
            ))}
            <hr className="invoice-divider" />
            <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>GST (5%)</span><span>{formatCurrency(Math.round((lastOrder.price / 1.05) * 0.05))}</span></div>
            <div className="invoice-row total"><span>TOTAL</span><strong style={{ color: 'var(--brand-primary)', fontSize: 20 }}>{formatCurrency(lastOrder.price)}</strong></div>
            <hr className="invoice-divider" />
            {lastOrder.paymentSession && (
              <>
                <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Payment Intent</span><span>{lastOrder.paymentSession.payment_intent_id}</span></div>
                <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Payment State</span><span>{lastOrder.paymentState}</span></div>
              </>
            )}
            {lastOrder.settlement && (
              <>
                <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Platform Fee</span><span>{formatCurrency(lastOrder.settlement.platform_fee)}</span></div>
                <div className="invoice-row"><span style={{ color: 'var(--text-muted)' }}>Restaurant Net</span><span>{formatCurrency(lastOrder.settlement.restaurant_net)}</span></div>
              </>
            )}
            <hr className="invoice-divider" />
            <div style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-muted)' }}>Thank you for dining with us! 🙏</div>
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setShowInvoice(false)}>Close</button>
              {lastOrder.paymentSession?.deeplink_or_qr && (
                <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => window.open(lastOrder.paymentSession.deeplink_or_qr, '_blank', 'noopener,noreferrer')}>
                  Pay Link
                </button>
              )}
              {lastOrder.paymentState !== 'PAID' && lastOrder.paymentSession && (
                <button className="btn btn-primary" style={{ flex: 1 }} onClick={handleMarkPaid}>
                  Mark Paid
                </button>
              )}
              <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => { window.print(); setShowInvoice(false) }}>
                <FileText size={14} /> Print
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── KDS View ─────────────────────────────────────────────────────────────────
function KDSView({ orders, upsertOrder }) {
  const buckets = {
    pending: orders.filter(o => o.status === 'Pending'),
    kitchen: orders.filter(o => ['Accepted', 'Preparing', 'In Kitchen'].includes(o.status)),
    ready: orders.filter(o => o.status === 'Ready'),
  }

  const move = async (order, next) => {
    if (!order.backendState) {
      toast.error('Excel history rows are read-only in KDS')
      return
    }
    try {
      const updated = await updateBackendOrderState(order.id, mapUiStateToBackend(next))
      upsertOrder(updated)
      toast.success(`${order.id} → ${next}`, { icon: '👨‍🍳' })
    } catch (error) {
      toast.error(error.message || 'Failed to update order')
    }
  }

  return (
    <div className="anim-fade kds-layout">
      <KDSColumn title="New Orders" items={buckets.pending} color="var(--status-pending)" renderActions={(order) => (
        <>
          <button className="btn btn-ghost btn-sm" style={{ flex: 1, fontSize: 11, color: 'var(--status-cancelled)' }} onClick={() => move(order, 'Cancelled')}>Reject</button>
          <button className="btn btn-primary btn-sm" style={{ flex: 1, fontSize: 11 }} onClick={() => move(order, 'Preparing')}>Accept</button>
        </>
      )} />
      <KDSColumn title="Preparing" items={buckets.kitchen} color="var(--status-kitchen)" renderActions={(order) => (
        <button className="btn btn-primary btn-sm" style={{ width: '100%', fontSize: 11 }} onClick={() => move(order, 'Ready')}>→ Ready</button>
      )} />
      <KDSColumn title="Ready" items={buckets.ready} color="var(--status-ready)" renderActions={(order) => (
        <button className="btn btn-primary btn-sm" style={{ width: '100%', fontSize: 11, background: 'var(--brand-green)' }} onClick={() => move(order, 'Delivered')}>✓ Handed Over</button>
      )} />
    </div>
  )
}

// ─── Tables View ──────────────────────────────────────────────────────────────
function TablesView({ tables: excelTables, orders, reservations }) {
  const [tables, setTables] = useState(excelTables)
  useEffect(() => { setTables(excelTables) }, [excelTables])

  const occupiedFromOrders = useMemo(() => {
    return orders
      .filter(o => ['Pending', 'In Kitchen', 'Preparing'].includes(o.status))
      .map(o => { const m = o.platform?.match(/T\d+/); return m ? m[0] : null })
      .filter(Boolean)
  }, [orders])

  const reservedFromRes = useMemo(() =>
    reservations.filter(r => r.status === 'Confirmed').map(r => r.table), [reservations])

  const sections = [...new Set(tables.map(t => t.section))]
  const stats = {
    available: tables.filter(t => !occupiedFromOrders.includes(t.id) && !reservedFromRes.includes(t.id)).length,
    occupied: occupiedFromOrders.length,
    reserved: reservedFromRes.length,
  }

  const toggle = (tableId) => {
    setTables(prev => prev.map(t => {
      if (t.id !== tableId) return t
      const next = t.status === 'Available' ? 'Occupied' : t.status === 'Occupied' ? 'Reserved' : 'Available'
      toast.success(`${tableId} → ${next}`)
      return { ...t, status: next }
    }))
  }

  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: 24 }}>
        <div className="metric-card"><div className="metric-icon green">🟢</div><div className="metric-value">{stats.available}</div><div className="metric-label">Available</div></div>
        <div className="metric-card"><div className="metric-icon orange">🔴</div><div className="metric-value">{stats.occupied}</div><div className="metric-label">Occupied (from orders)</div></div>
        <div className="metric-card"><div className="metric-icon gold">🟡</div><div className="metric-value">{stats.reserved}</div><div className="metric-label">Reserved (from sheet)</div></div>
      </div>

      {sections.map(section => (
        <div key={section} style={{ marginBottom: 24 }}>
          <div className="section-header"><div className="section-title">{section}</div></div>
          <div className="tables-grid">
            {tables.filter(t => t.section === section).map(table => {
              const isOccupied = occupiedFromOrders.includes(table.id)
              const isReserved = reservedFromRes.includes(table.id)
              const statusClass = isOccupied ? 'occupied' : isReserved ? 'reserved' : table.status.toLowerCase()
              const statusLabel = isOccupied ? 'Occupied' : isReserved ? 'Reserved' : table.status
              return (
                <div key={table.id} className={`table-cell ${statusClass}`} onClick={() => toggle(table.id)} title={`Click to cycle status`}>
                  <div className="table-num">{table.id}</div>
                  <div className="table-cap">{table.capacity} pax</div>
                  <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.7, fontWeight: 700 }}>{statusLabel}</div>
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Reservations View ────────────────────────────────────────────────────────
function ReservationsView({ reservations }) {
  const statusColors = { Confirmed: 'badge-ready', Pending: 'badge-pending', Cancelled: 'badge-cancelled' }
  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: 20 }}>
        <div className="metric-card"><div className="metric-icon green">✅</div><div className="metric-value">{reservations.filter(r => r.status === 'Confirmed').length}</div><div className="metric-label">Confirmed</div></div>
        <div className="metric-card"><div className="metric-icon gold">⏳</div><div className="metric-value">{reservations.filter(r => r.status === 'Pending').length}</div><div className="metric-label">Pending</div></div>
        <div className="metric-card"><div className="metric-icon blue">👥</div><div className="metric-value">{reservations.reduce((s, r) => s + r.guests, 0)}</div><div className="metric-label">Total Guests Expected</div></div>
      </div>

      <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead><tr>
              <th>Res ID</th><th>Customer</th><th>Phone</th>
              <th>Date</th><th>Time</th><th>Guests</th>
              <th>Table</th><th>Status</th><th>Requests</th>
            </tr></thead>
            <tbody>
              {reservations.length === 0
                ? <tr><td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>No reservations in Excel sheet</td></tr>
                : reservations.map(r => (
                  <tr key={r.id}>
                    <td><strong style={{ fontSize: 12 }}>{r.id}</strong></td>
                    <td><div style={{ fontWeight: 600 }}>{r.customer}</div><div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{r.email}</div></td>
                    <td style={{ fontSize: 12 }}>{r.phone}</td>
                    <td style={{ fontSize: 12 }}>{r.date}</td>
                    <td style={{ fontSize: 12 }}>{r.time}</td>
                    <td><span className="badge badge-kitchen">{r.guests} pax</span></td>
                    <td><span className="badge badge-special">{r.table}</span></td>
                    <td><span className={`badge ${statusColors[r.status] || 'badge-pending'}`}>{r.status}</span></td>
                    <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 160 }}>{r.requests || '—'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Orders View ──────────────────────────────────────────────────────────────
function OrdersView({ orders, upsertOrder, globalSearch }) {
  const [statusFilter, setStatusFilter] = useState('All')
  const [platformFilter, setPlatformFilter] = useState('All')

  const filtered = useMemo(() =>
    orders.filter(o => {
      const q = globalSearch.trim().toLowerCase()
      const matchQ = !q || [o.id, o.customer, o.phone, o.item].some(f => String(f).toLowerCase().includes(q))
      const matchS = statusFilter === 'All' || o.status === statusFilter
      const matchP = platformFilter === 'All' || String(o.platform).toLowerCase().includes(platformFilter.toLowerCase())
      return matchQ && matchS && matchP
    }),
    [orders, globalSearch, statusFilter, platformFilter]
  )

  const platforms = [...new Set(orders.map(o => o.platform).filter(Boolean))]

  return (
    <div className="anim-fade">
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        <select className="form-input" style={{ width: 'auto' }} value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          <option value="All">All Status</option>
          {STATUS_OPTIONS.map(s => <option key={s}>{s}</option>)}
          <option>Preparing</option>
        </select>
        <select className="form-input" style={{ width: 'auto' }} value={platformFilter} onChange={e => setPlatformFilter(e.target.value)}>
          <option value="All">All Platforms</option>
          {platforms.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <div style={{ marginLeft: 'auto', fontSize: 13, color: 'var(--text-muted)' }}>
          {filtered.length} of {orders.length} orders
        </div>
      </div>

      <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead><tr>
              <th>Order ID</th><th>Customer</th><th>Items</th>
              <th>Platform</th><th>Amount</th><th>Date/Time</th>
              <th>ETA</th><th>Status</th>
            </tr></thead>
            <tbody>
              {filtered.length === 0
                ? <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>No orders found</td></tr>
                : filtered.map(order => (
                  <tr key={order.id}>
                    <td><strong style={{ fontSize: 12 }}>{order.id}</strong></td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{order.customer}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{order.phone}</div>
                    </td>
                    <td style={{ maxWidth: 200 }}>
                      <div style={{ fontSize: 12, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{order.item}</div>
                      {order.special && <div style={{ fontSize: 10, color: 'var(--brand-primary)' }}>Note: {order.special}</div>}
                    </td>
                    <td><span className="badge badge-special" style={{ fontSize: 10 }}>{order.platform}</span></td>
                    <td><strong>{formatCurrency(order.price)}</strong></td>
                    <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{order.dateTime}</td>
                    <td style={{ fontSize: 12 }}>{order.eta}</td>
                    <td>
                      <select className="order-status-select" value={order.status}
                        onChange={async e => {
                          const next = e.target.value
                          if (!order.backendState) {
                            toast.error('Excel history rows are read-only')
                            return
                          }
                          try {
                            const updated = await updateBackendOrderState(order.id, mapUiStateToBackend(next))
                            upsertOrder(updated)
                          } catch (error) {
                            toast.error(error.message || 'Failed to update order')
                          }
                        }}>
                        {[...STATUS_OPTIONS, 'Preparing'].map(s => <option key={s}>{s}</option>)}
                      </select>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Invoices View ────────────────────────────────────────────────────────────
function InvoicesView({ invoices }) {
  return (
    <div className="anim-fade">
      <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead><tr><th>Order ID</th><th>Customer</th><th>Date/Time</th><th>Amount</th><th>Status</th><th>Invoice Link</th></tr></thead>
            <tbody>
              {invoices.length === 0
                ? <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>No invoices in Excel sheet</td></tr>
                : invoices.map((inv, i) => (
                  <tr key={i}>
                    <td><strong style={{ fontSize: 12 }}>{inv.orderId}</strong></td>
                    <td>{inv.customer}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{inv.dateTime}</td>
                    <td><strong>{formatCurrency(inv.amount)}</strong></td>
                    <td><span className="badge badge-ready">{inv.status}</span></td>
                    <td>
                      {inv.url
                        ? <a href={inv.url} target="_blank" rel="noreferrer" style={{ color: 'var(--brand-primary)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
                            View <ArrowUpRight size={12} />
                          </a>
                        : <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>—</span>}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Menu Manager View ────────────────────────────────────────────────────────
function MenuManagerView({ menu }) {
  const cats = ['All', ...new Set(menu.map(m => m.category))]
  const [category, setCategory] = useState('All')
  const [search, setSearch] = useState('')

  const filtered = menu.filter(item => {
    const catOk = category === 'All' || item.category === category
    const searchOk = !search || item.name.toLowerCase().includes(search.toLowerCase())
    return catOk && searchOk
  })

  return (
    <div className="anim-fade">
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="search-bar" style={{ flex: '0 0 240px', margin: 0 }}>
          <Search size={13} color="var(--text-muted)" />
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search menu…" />
        </div>
        <div className="category-pills">
          {cats.map(cat => (
            <button key={cat} className={`cat-pill ${category === cat ? 'active' : ''}`} onClick={() => setCategory(cat)}>{cat}</button>
          ))}
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 13, color: 'var(--text-muted)' }}>{filtered.length} of {menu.length} items · From <strong style={{ color: 'var(--brand-primary)' }}>menu card</strong> sheet</div>
      </div>

      <div className="menu-management-grid">
        {filtered.map(item => (
          <div key={item.id} className="menu-mgmt-card">
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ fontSize: 28 }}>{item.emoji}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2, flexWrap: 'wrap' }}>
                  <div className={`veg-dot ${item.category === 'Non-Veg' ? 'nonveg' : 'veg'}`} />
                  <div style={{ fontWeight: 700, fontSize: 14 }}>{item.name}</div>
                  {item.isSpecial && <span className="badge badge-special" style={{ fontSize: 9 }}>🔥 Special</span>}
                  {item.stock <= 3 && <span className="badge badge-cancelled" style={{ fontSize: 9 }}>Low Stock</span>}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>{item.description}</div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 800, color: 'var(--brand-primary)', fontSize: 15 }}>
                    {formatCurrency(item.isSpecial && item.specialPrice ? item.specialPrice : item.price)}
                  </span>
                  {item.isSpecial && item.specialPrice && (
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', textDecoration: 'line-through' }}>{formatCurrency(item.price)}</span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                  <span>📦 Stock: {item.stock}</span>
                  <span>⏱ {item.prepTime}</span>
                  <span>ID: {item.id}</span>
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 12 }}>
              <button className="btn btn-ghost btn-sm" style={{ flex: 1, fontSize: 11 }}>Edit</button>
              <span className={`badge ${item.status === 'Available' ? 'badge-ready' : 'badge-cancelled'}`} style={{ padding: '6px 12px' }}>
                {item.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Specials View ────────────────────────────────────────────────────────────
function SpecialsView({ specials }) {
  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: 20 }}>
        <div className="metric-card"><div className="metric-icon orange">🔥</div><div className="metric-value">{specials.length}</div><div className="metric-label">Active Specials</div></div>
        <div className="metric-card"><div className="metric-icon green">💸</div><div className="metric-value">Up to {Math.max(...specials.map(s => s.discount), 0)}%</div><div className="metric-label">Max Discount</div></div>
        <div className="metric-card"><div className="metric-icon gold">📊</div><div className="metric-value">{specials.length > 0 ? Math.round(specials.reduce((s, i) => s + i.discount, 0) / specials.length) : 0}%</div><div className="metric-label">Avg Discount</div></div>
      </div>

      <div className="section-header">
        <div>
          <div className="section-title">Special Offers</div>
          <div className="section-sub">From <strong style={{ color: 'var(--brand-primary)' }}>Special items</strong> Excel sheet</div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {specials.map(item => {
          const discounted = Math.round(item.normalPrice * (1 - item.discount / 100))
          return (
            <div key={item.id} className="offer-card">
              <div className="offer-discount-badge">-{item.discount}%</div>
              <div style={{ fontSize: 24 }}>{item.emoji}</div>
              <div className="offer-info">
                <div className="offer-name">{item.name}</div>
                <div className="offer-category">{item.category} · ID: {item.itemId}</div>
              </div>
              <div className="offer-prices">
                <div className="offer-new-price">{formatCurrency(discounted)}</div>
                <div className="offer-old-price">{formatCurrency(item.normalPrice)}</div>
              </div>
            </div>
          )
        })}
        {specials.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 60, fontSize: 14 }}>No specials in Excel sheet</div>
        )}
      </div>
    </div>
  )
}

// ─── Employees View ───────────────────────────────────────────────────────────
function EmployeesView({ employees }) {
  const deptColors = { Management: '#FF6B35', Kitchen: '#1A936F', 'Front Desk': '#004E89', Delivery: '#F59E0B', Service: '#7B2FBE' }
  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 20 }}>
        <div className="metric-card"><div className="metric-icon blue">👥</div><div className="metric-value">{employees.length}</div><div className="metric-label">Total Staff</div></div>
        <div className="metric-card"><div className="metric-icon green">✅</div><div className="metric-value">{employees.filter(e => e.status === 'Active').length}</div><div className="metric-label">Active</div></div>
        <div className="metric-card"><div className="metric-icon orange">⭐</div><div className="metric-value">{employees.length > 0 ? (employees.reduce((s, e) => s + e.rating, 0) / employees.length).toFixed(1) : 0}</div><div className="metric-label">Avg Rating</div></div>
        <div className="metric-card"><div className="metric-icon purple">💰</div><div className="metric-value">{formatCurrency(employees.reduce((s, e) => s + e.salary, 0))}</div><div className="metric-label">Total Payroll</div></div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {employees.map(emp => (
          <div key={emp.id} className="menu-mgmt-card">
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{
                width: 44, height: 44, borderRadius: '50%',
                background: `${deptColors[emp.dept] || '#7B2FBE'}22`,
                border: `2px solid ${deptColors[emp.dept] || '#7B2FBE'}44`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 20, flexShrink: 0
              }}>
                {emp.dept === 'Kitchen' ? '👨‍🍳' : emp.dept === 'Management' ? '👔' : emp.dept === 'Delivery' ? '🛵' : '🧑‍💼'}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, fontSize: 14 }}>{emp.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{emp.role}</div>
                <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                  <span className="badge badge-special" style={{ fontSize: 10 }}>{emp.dept}</span>
                  <span className="badge badge-kitchen" style={{ fontSize: 10 }}>{emp.shift}</span>
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--brand-primary)' }}>{formatCurrency(emp.salary)}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>/month</div>
                <div style={{ fontSize: 12, color: '#F59E0B', marginTop: 4 }}>
                  {'★'.repeat(Math.round(emp.rating))} {emp.rating}
                </div>
              </div>
            </div>
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
              Skills: {emp.skills || '—'}
            </div>
          </div>
        ))}
        {employees.length === 0 && (
          <div style={{ gridColumn: '1/-1', textAlign: 'center', color: 'var(--text-muted)', padding: 60, fontSize: 14 }}>No employees in Excel sheet</div>
        )}
      </div>
    </div>
  )
}

// ─── Analytics View ───────────────────────────────────────────────────────────
function AnalyticsView({ analytics, salesByItem }) {
  const totalRevenue = analytics.reduce((s, d) => s + d.revenue, 0)
  const totalOrders = analytics.reduce((s, d) => s + d.orders, 0)
  const totalDelivery = analytics.reduce((s, d) => s + d.delivery, 0)

  return (
    <div className="anim-fade">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 20 }}>
        <div className="metric-card"><div className="metric-icon orange">💰</div><div className="metric-value">{formatCurrency(totalRevenue)}</div><div className="metric-label">Total Revenue (period)</div></div>
        <div className="metric-card"><div className="metric-icon blue">📋</div><div className="metric-value">{totalOrders}</div><div className="metric-label">Total Orders</div></div>
        <div className="metric-card"><div className="metric-icon green">🛵</div><div className="metric-value">{totalDelivery}</div><div className="metric-label">Delivery Orders</div></div>
        <div className="metric-card"><div className="metric-icon purple">📊</div><div className="metric-value">{formatCurrency(totalOrders > 0 ? Math.round(totalRevenue / totalOrders) : 0)}</div><div className="metric-label">Overall Avg Order</div></div>
      </div>

      <div className="chart-card" style={{ marginBottom: 20 }}>
        <div className="chart-header">
          <div><div className="section-title">Revenue by Day</div><div className="section-sub">Source: Order Analytical sheet</div></div>
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={analytics}>
            <defs>
              <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#FF6B35" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#FF6B35" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#888', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={v => `Rs. ${(v / 1000).toFixed(0)}K`} />
            <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF' }} formatter={v => [formatCurrency(v), 'Revenue']} />
            <Area type="monotone" dataKey="revenue" stroke="#FF6B35" strokeWidth={3} fill="url(#g1)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Orders per day */}
        <div className="chart-card">
          <div className="chart-header"><div className="section-title">Orders per Day</div></div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={analytics} barSize={22}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF' }} />
              <Bar dataKey="orders" fill="#1A936F" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Avg order value */}
        <div className="chart-card">
          <div className="chart-header"><div className="section-title">Avg Order Value</div></div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={analytics}>
              <defs>
                <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#7B2FBE" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#7B2FBE" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => `Rs. ${v}`} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, color: '#F0F0FF' }} formatter={v => [formatCurrency(v), 'Avg']} />
              <Area type="monotone" dataKey="avg" stroke="#7B2FBE" strokeWidth={2.5} fill="url(#g2)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top items from Sales Dashboard sheet */}
      {salesByItem.length > 0 && (
        <div className="chart-card" style={{ marginTop: 20 }}>
          <div className="chart-header">
            <div><div className="section-title">Top Items by Quantity</div><div className="section-sub">Source: Sales Dashboard sheet</div></div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {salesByItem.slice(0, 8).map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 24, height: 24, borderRadius: 6, background: 'var(--bg-overlay)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 800, color: 'var(--brand-primary)' }}>{i + 1}</div>
                <div style={{ flex: 1, fontSize: 13, color: 'var(--text-secondary)' }}>{item.item}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 80, height: 6, background: 'var(--bg-overlay)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, (item.qty / (salesByItem[0]?.qty || 1)) * 100)}%`, height: '100%', background: 'var(--brand-primary)', borderRadius: 3 }} />
                  </div>
                  <span style={{ fontSize: 13, fontWeight: 700, minWidth: 40, textAlign: 'right' }}>{item.qty} qty</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const { data, loading, error, reload, lastLoaded } = useExcelData()
  const [activeView, setActiveView] = useState('dashboard')
  const [globalSearch, setGlobalSearch] = useState('')
  const [backendOrders, setBackendOrders] = useState([])
  const safeData = data || {
    menu: [],
    analytics: [],
    specials: [],
    tables: [],
    reservations: [],
    employees: [],
    invoices: [],
    salesByItem: [],
    orders: [],
  }
  const { menu, analytics, specials, tables, reservations, employees, invoices, salesByItem, orders: excelOrders } = safeData
  const liveOrders = useMemo(() => {
    const merged = new Map()
    backendOrders.forEach(order => merged.set(order.id, order))
    excelOrders.forEach(order => {
      if (!merged.has(order.id)) {
        merged.set(order.id, order)
      }
    })
    return Array.from(merged.values()).sort((a, b) => String(b.id).localeCompare(String(a.id)))
  }, [backendOrders, excelOrders])

  async function refreshBackendOrders({ silent } = { silent: false }) {
    try {
      const orders = await fetchBackendOrders()
      startTransition(() => {
        setBackendOrders(orders)
      })
    } catch (backendError) {
      if (!silent) {
        toast.error(backendError.message || 'Backend sync failed')
      }
    }
  }

  function upsertOrder(order) {
    setBackendOrders(prev => {
      const next = prev.filter(existing => existing.id !== order.id)
      return [order, ...next]
    })
  }

  useEffect(() => {
    let cancelled = false
    fetchBackendOrders()
      .then(orders => {
        if (!cancelled) {
          startTransition(() => {
            setBackendOrders(orders)
          })
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  if (loading) return <LoadingScreen />
  if (error) return <ErrorScreen error={error} onRetry={reload} />

  const pendingCount = liveOrders.filter(o => o.status === 'Pending').length

  function renderView() {
    switch (activeView) {
      case 'dashboard':    return <DashboardView analytics={analytics} menu={menu} orders={liveOrders} specials={specials} />
      case 'pos':          return <POSView menu={menu} tables={tables} upsertOrder={upsertOrder} />
      case 'kds':          return <KDSView orders={liveOrders} upsertOrder={upsertOrder} />
      case 'tables':       return <TablesView tables={tables} orders={liveOrders} reservations={reservations} />
      case 'reservations': return <ReservationsView reservations={reservations} />
      case 'orders':       return <OrdersView orders={liveOrders} upsertOrder={upsertOrder} globalSearch={globalSearch} />
      case 'invoices':     return <InvoicesView invoices={invoices} />
      case 'menu':         return <MenuManagerView menu={menu} />
      case 'specials':     return <SpecialsView specials={specials} />
      case 'employees':    return <EmployeesView employees={employees} />
      case 'analytics':    return <AnalyticsView analytics={analytics} salesByItem={salesByItem} />
      default:             return <DashboardView analytics={analytics} menu={menu} orders={liveOrders} specials={specials} />
    }
  }

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#1A1A28', color: '#F0F0FF',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 12, fontSize: 13, fontFamily: 'Inter, sans-serif',
          },
          success: { iconTheme: { primary: '#1A936F', secondary: '#F0F0FF' } },
          error: { iconTheme: { primary: '#EF4444', secondary: '#F0F0FF' } },
        }}
      />
      <div className="app-layout">
        <Sidebar active={activeView} setActive={setActiveView} pendingCount={pendingCount} lastLoaded={lastLoaded} />
        <div className="main-area">
          <Topbar
            activeView={activeView} pendingCount={pendingCount}
            globalSearch={globalSearch} setGlobalSearch={setGlobalSearch}
            onRefresh={async () => {
              reload()
              await refreshBackendOrders()
              toast.success('Excel and backend reloaded!')
            }}
          />
          <div className="content">{renderView()}</div>
        </div>
      </div>
    </>
  )
}
