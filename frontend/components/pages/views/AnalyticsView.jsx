'use client'

import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Clock3, IndianRupee, Loader2, RefreshCw, Send } from 'lucide-react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchOperationsAnalytics } from '../../../services/platformApi'

const empty = {
  range: { from: '', to: '' }, scheduled_volume: [],
  lifecycle: { scheduled: 0, submitted: 0, completed: 0, terminal_failures: 0, submission_success_rate: 0, fulfillment_success_rate: 0 },
  scheduler: { attempts: 0, average_lag_seconds: 0, dead_letter_count: 0, missed_window_count: 0 },
  providers: { by_status: {}, failure_count: 0, unknown_count: 0, failure_rate: 0, unknown_rate: 0 },
  completed_revenue: [], fulfillment_performance: [], reasons: { cancellations: [], reschedules: [] },
}

const money = cents => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(cents || 0) / 100)
const duration = seconds => seconds < 60 ? `${Math.round(seconds)}s` : `${Math.round(seconds / 60)}m`
const label = value => String(value || '').replaceAll('_', ' ').toLowerCase().replace(/^./, char => char.toUpperCase())

function Metric({ icon: Icon, label: name, value, detail, tone = '' }) {
  return <article className={`oa-metric ${tone}`}><span><Icon size={17} /></span><div><small>{name}</small><strong>{value}</strong><p>{detail}</p></div></article>
}

export default function AnalyticsView() {
  const [data, setData] = useState(empty)
  const [filters, setFilters] = useState({ from: '', to: '' })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async (from = '', to = '') => {
    setLoading(true); setError('')
    try {
      const result = await fetchOperationsAnalytics(from, to)
      setData(result)
      setFilters({ from: result.range.from, to: result.range.to })
    } catch (err) {
      setError(err.message || 'Analytics could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const scheduledVolume = data.scheduled_volume.map(row => ({
    time: new Date(row.desired_hour).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: 'numeric' }),
    meals: row.count,
  }))
  const revenue = data.completed_revenue.map(row => ({
    date: new Date(`${row.fulfillment_date}T12:00:00`).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }),
    revenue: row.amount_cents / 100,
  }))
  const totalRevenue = data.completed_revenue.reduce((sum, row) => sum + row.amount_cents, 0)

  if (loading && !data.range.from) return <div className="oa-loading"><Loader2 className="spin" size={22} /> Building operational analytics…</div>

  return <div className="oa-view">
    <div className="oa-toolbar">
      <div><span>FULFILLMENT INTELLIGENCE</span><h3>Performance by the timestamp that matters.</h3><p>Desired time measures demand; submission and completion measure execution.</p></div>
      <form onSubmit={event => { event.preventDefault(); load(filters.from, filters.to) }}>
        <label>From<input type="date" value={filters.from} onChange={event => setFilters({ ...filters, from: event.target.value })} /></label>
        <label>To<input type="date" value={filters.to} onChange={event => setFilters({ ...filters, to: event.target.value })} /></label>
        <button disabled={loading}>{loading ? <Loader2 className="spin" size={14} /> : <RefreshCw size={14} />} Apply</button>
      </form>
    </div>

    {error && <div className="oa-error"><AlertTriangle size={16} />{error}</div>}

    <section className="oa-metrics">
      <Metric icon={Send} label="Submission success" value={`${data.lifecycle.submission_success_rate}%`} detail={`${data.lifecycle.submitted} of ${data.lifecycle.scheduled} scheduled`} />
      <Metric icon={CheckCircle2} label="Fulfillment success" value={`${data.lifecycle.fulfillment_success_rate}%`} detail={`${data.lifecycle.completed} completed`} tone="green" />
      <Metric icon={Clock3} label="Average scheduler lag" value={duration(data.scheduler.average_lag_seconds)} detail={`${data.scheduler.missed_window_count} missed windows`} tone={data.scheduler.missed_window_count ? 'warn' : ''} />
      <Metric icon={IndianRupee} label="Completed revenue" value={money(totalRevenue)} detail="Immutable ledger, by fulfillment date" tone="green" />
    </section>

    <section className="oa-chart-grid">
      <article className="oa-card"><header><div><small>DEMAND</small><h4>Scheduled volume by desired time</h4></div><span>{data.lifecycle.scheduled} meals</span></header>
        {scheduledVolume.length ? <ResponsiveContainer width="100%" height={260}><BarChart data={scheduledVolume}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.07)" /><XAxis dataKey="time" tick={{ fontSize: 10, fill: '#8f9099' }} minTickGap={28} /><YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#8f9099' }} /><Tooltip contentStyle={{ background: '#171821', border: '1px solid #30313d', borderRadius: 10 }} /><Bar dataKey="meals" fill="#dd7048" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer> : <p className="oa-empty">No scheduled meals in this range.</p>}
      </article>
      <article className="oa-card"><header><div><small>REVENUE</small><h4>Completed revenue by fulfillment date</h4></div><span>{money(totalRevenue)}</span></header>
        {revenue.length ? <ResponsiveContainer width="100%" height={260}><AreaChart data={revenue}><defs><linearGradient id="phase7Revenue" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#1a936f" stopOpacity={0.4} /><stop offset="95%" stopColor="#1a936f" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.07)" /><XAxis dataKey="date" tick={{ fontSize: 10, fill: '#8f9099' }} /><YAxis tick={{ fontSize: 10, fill: '#8f9099' }} /><Tooltip formatter={value => [`₹${Number(value).toLocaleString('en-IN')}`, 'Revenue']} contentStyle={{ background: '#171821', border: '1px solid #30313d', borderRadius: 10 }} /><Area dataKey="revenue" stroke="#1a936f" fill="url(#phase7Revenue)" strokeWidth={2} /></AreaChart></ResponsiveContainer> : <p className="oa-empty">No completed revenue in this range.</p>}
      </article>
    </section>

    <section className="oa-detail-grid">
      <article className="oa-card"><header><div><small>FULFILLMENT MODES</small><h4>Delivery, pickup & drive-through</h4></div></header>
        <div className="oa-table"><div className="oa-table-head"><span>Mode</span><span>Scheduled</span><span>Completed</span><span>Success</span><span>Avg. time</span></div>
          {data.fulfillment_performance.map(row => <div className="oa-table-row" key={row.type}><strong>{label(row.type)}</strong><span>{row.scheduled}</span><span>{row.completed}</span><span>{row.success_rate}%</span><span>{duration(row.average_seconds_from_submission)}</span></div>)}
          {!data.fulfillment_performance.length && <p className="oa-empty">No fulfillment activity.</p>}
        </div>
      </article>
      <article className="oa-card"><header><div><small>RELIABILITY</small><h4>Scheduler & provider health</h4></div></header>
        <div className="oa-health-list">
          <div><span>Scheduler attempts</span><strong>{data.scheduler.attempts}</strong></div>
          <div><span>Dead-letter jobs</span><strong className={data.scheduler.dead_letter_count ? 'danger' : ''}>{data.scheduler.dead_letter_count}</strong></div>
          <div><span>Provider failures</span><strong className={data.providers.failure_count ? 'danger' : ''}>{data.providers.failure_count} · {data.providers.failure_rate}%</strong></div>
          <div><span>Unknown provider outcomes</span><strong className={data.providers.unknown_count ? 'warn' : ''}>{data.providers.unknown_count} · {data.providers.unknown_rate}%</strong></div>
        </div>
      </article>
    </section>

    <section className="oa-detail-grid">
      <article className="oa-card"><header><div><small>CANCELLATIONS</small><h4>Recorded reasons</h4></div></header><div className="oa-reasons">{data.reasons.cancellations.map(item => <span key={item.reason}>{label(item.reason)} <strong>{item.count}</strong></span>)}{!data.reasons.cancellations.length && <p className="oa-empty">No cancellations in this range.</p>}</div></article>
      <article className="oa-card"><header><div><small>RESCHEDULES</small><h4>Recorded reasons</h4></div></header><div className="oa-reasons">{data.reasons.reschedules.map(item => <span key={item.reason}>{label(item.reason)} <strong>{item.count}</strong></span>)}{!data.reasons.reschedules.length && <p className="oa-empty">No reschedules in this range.</p>}</div></article>
    </section>
  </div>
}
