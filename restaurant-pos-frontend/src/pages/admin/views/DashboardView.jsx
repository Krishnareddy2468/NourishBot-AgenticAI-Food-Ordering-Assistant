import { useMemo } from 'react'
import { TrendingUp, TrendingDown, ShoppingCart, Users, Clock, CheckCircle2 } from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'

function StatCard({ label, value, sub, trend, color }) {
  const Icon = trend === 'up' ? TrendingUp : TrendingDown
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      {sub && (
        <div className="stat-sub" style={{ color: trend === 'up' ? 'var(--brand-green)' : 'var(--status-cancelled)' }}>
          {trend && <Icon size={12} />} {sub}
        </div>
      )}
    </div>
  )
}

export default function DashboardView({ orders }) {
  const stats = useMemo(() => {
    const revenue = orders.reduce((s, o) => s + (o.price || 0), 0)
    const pending = orders.filter(o => o.status === 'Pending').length
    const delivered = orders.filter(o => o.status === 'Delivered').length
    return { total: orders.length, revenue, pending, delivered }
  }, [orders])

  // Build hourly chart data from orders
  const chartData = useMemo(() => {
    const hours = Array.from({ length: 12 }, (_, i) => ({
      hour: `${(i + 10) % 12 || 12}${i + 10 < 12 ? 'AM' : 'PM'}`,
      orders: 0,
      revenue: 0,
    }))
    orders.forEach(o => {
      const h = o.createdAt ? new Date(o.createdAt).getHours() : -1
      const idx = h - 10
      if (idx >= 0 && idx < 12) {
        hours[idx].orders++
        hours[idx].revenue += o.price || 0
      }
    })
    return hours
  }, [orders])

  const byStatus = [
    { name: 'Pending', value: orders.filter(o => o.status === 'Pending').length, fill: '#F59E0B' },
    { name: 'Preparing', value: orders.filter(o => o.status === 'Preparing').length, fill: '#3B82F6' },
    { name: 'Ready', value: orders.filter(o => o.status === 'Ready').length, fill: '#8B5CF6' },
    { name: 'Delivered', value: orders.filter(o => o.status === 'Delivered').length, fill: '#1A936F' },
    { name: 'Cancelled', value: orders.filter(o => o.status === 'Cancelled').length, fill: '#EF4444' },
  ]

  return (
    <div className="dashboard-view">
      <div className="stat-grid">
        <StatCard label="Total Orders" value={stats.total} color="var(--brand-primary)" />
        <StatCard label="Revenue" value={`₹${stats.revenue.toLocaleString('en-IN')}`} color="var(--brand-green)" />
        <StatCard label="Pending" value={stats.pending} color="#F59E0B" />
        <StatCard label="Delivered" value={stats.delivered} color="var(--brand-green)" />
      </div>

      <div className="dashboard-charts">
        <div className="chart-card">
          <h3>Orders by Hour</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="orderGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF6B35" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#FF6B35" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="hour" tick={{ fontSize: 11, fill: '#888' }} />
              <YAxis tick={{ fontSize: 11, fill: '#888' }} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <Area type="monotone" dataKey="orders" stroke="#FF6B35" fill="url(#orderGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Orders by Status</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byStatus}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#888' }} />
              <YAxis tick={{ fontSize: 11, fill: '#888' }} />
              <Tooltip contentStyle={{ background: '#1A1A28', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <Bar dataKey="value" fill="#FF6B35" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
