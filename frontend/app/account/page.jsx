'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import {
  AlertCircle, ArrowLeft, ArrowRight, CalendarClock, CalendarDays, Check, Clock3,
  IndianRupee, Loader2, LogOut, Pause, Play, RefreshCw, RotateCcw,
  Settings2, ShieldCheck, ShoppingBag, SkipForward, WalletCards,
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import NourishLogo from '../../components/shared/NourishLogo'
import {
  applyCustomerRoutineAction,
  cancelCustomerOrder,
  confirmCustomerMealPlan,
  fetchCustomerDashboard,
  updateCustomerBudget,
  updateCustomerConsent,
  updateCustomerPreferences,
} from '../../services/platformApi'

const emptyDashboard = {
  week: { starts_on: '', ends_on: '', days: [] }, upcoming_confirmed: [], draft_plans: [],
  templates: [], routines: [], orders: [], provider_attempts: [],
  preferences: { dietary_flags: [], allergies: [], health_goals: [], price_band: 'mid' },
  consents: {}, budgets: [], spending: { week_cents: 0, lifetime_cents: 0, currency: 'INR' },
  provider_capabilities: [],
}

const money = (cents = 0, currency = 'INR') => new Intl.NumberFormat('en-IN', {
  style: 'currency', currency, maximumFractionDigits: 0,
}).format(Number(cents) / 100)

const pretty = value => String(value || '').replaceAll('_', ' ').toLowerCase().replace(/^./, char => char.toUpperCase())
const splitTerms = value => value.split(',').map(term => term.trim()).filter(Boolean)
const localDate = value => new Date(`${value}T12:00:00`).toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })
const localDateTime = value => value ? new Date(value).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' }) : 'Not recorded'
const dateTimeInput = value => {
  const date = new Date(value)
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16)
}

function shiftWeek(start, amount) {
  const date = start ? new Date(`${start}T12:00:00`) : new Date()
  date.setDate(date.getDate() + amount)
  return date.toISOString().slice(0, 10)
}

export default function AccountPage() {
  const { user, role, isLoggedIn, authLoading, logout } = useAuth()
  const router = useRouter()
  const [data, setData] = useState(emptyDashboard)
  const [selectedDate, setSelectedDate] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [preferenceForm, setPreferenceForm] = useState({ dietary_flags: '', allergies: '', health_goals: '', price_band: 'mid' })
  const [budgetForm, setBudgetForm] = useState({ period: 'WEEKLY', amount: '', starts_on: '' })
  const [rescheduleTimes, setRescheduleTimes] = useState({})

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (!authLoading && role && role !== 'CUSTOMER') router.replace('/workspace')
  }, [authLoading, isLoggedIn, role, router])

  const load = useCallback(async (weekStart = '') => {
    setLoading(true)
    setError('')
    try {
      const result = await fetchCustomerDashboard(weekStart)
      setData(result)
      setSelectedDate(current => result.week.days.some(day => day.date === current) ? current : result.week.days[0]?.date || '')
      setPreferenceForm({
        dietary_flags: result.preferences.dietary_flags.join(', '),
        allergies: result.preferences.allergies.join(', '),
        health_goals: result.preferences.health_goals.join(', '),
        price_band: result.preferences.price_band,
      })
      const activeBudget = result.budgets[0]
      setBudgetForm({
        period: activeBudget?.period || 'WEEKLY',
        amount: activeBudget ? String(activeBudget.limit_cents / 100) : '',
        starts_on: activeBudget?.starts_on || result.week.starts_on,
      })
    } catch (err) {
      setError(err.message || 'Your dashboard could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!authLoading && isLoggedIn && role === 'CUSTOMER') load()
  }, [authLoading, isLoggedIn, role, load])

  const run = async (key, action, success) => {
    setBusy(key); setError(''); setNotice('')
    try {
      await action()
      setNotice(success)
      await load(data.week.starts_on)
    } catch (err) {
      setError(err.message || 'The change could not be saved.')
    } finally {
      setBusy('')
    }
  }

  const selectedDay = data.week.days.find(day => day.date === selectedDate) || { meals: [] }
  const nextMeal = data.upcoming_confirmed[0]
  const activeBudget = data.budgets[0]
  const budgetPercent = activeBudget ? Math.min(100, Math.round(data.spending.week_cents / activeBudget.limit_cents * 100)) : 0
  const attemptsByOrder = useMemo(() => data.provider_attempts.reduce((result, attempt) => {
    result[attempt.order_id] = result[attempt.order_id] || []
    result[attempt.order_id].push(attempt)
    return result
  }, {}), [data.provider_attempts])

  if (authLoading || !isLoggedIn || role !== 'CUSTOMER') return null

  async function signOut() {
    await logout()
    router.replace('/login')
  }

  async function savePreferences(event) {
    event.preventDefault()
    await run('preferences', () => updateCustomerPreferences({
      dietary_flags: splitTerms(preferenceForm.dietary_flags),
      allergies: splitTerms(preferenceForm.allergies),
      health_goals: splitTerms(preferenceForm.health_goals),
      price_band: preferenceForm.price_band,
    }), 'Preferences updated.')
  }

  async function saveBudget(event) {
    event.preventDefault()
    await run('budget', () => updateCustomerBudget({
      period: budgetForm.period,
      limit_cents: Math.round(Number(budgetForm.amount) * 100),
      currency: 'INR', starts_on: budgetForm.starts_on, alert_thresholds: [50, 80, 100],
    }), 'Budget limit updated.')
  }

  const firstName = user?.name?.split(' ')[0] || 'there'

  return (
    <main className="customer-account-page ca-page">
      <header className="ca-nav">
        <NourishLogo className="ca-logo" />
        <nav className="ca-nav-links" aria-label="Customer navigation">
          <a className="active" href="#planner">Planner</a><a href="#routines">Routines</a>
          <a href="#preferences">Preferences</a><a href="#activity">Activity</a>
        </nav>
        <div className="ca-nav-actions">
          <span className="ca-user-dot" aria-hidden="true">{firstName.charAt(0).toUpperCase()}</span>
          <button type="button" onClick={signOut}><LogOut size={15} /><span>Sign out</span></button>
        </div>
      </header>

      <div className="ca-shell">
        <section className="ca-welcome">
          <div><span className="ca-eyebrow">YOUR MEAL SPACE</span><h1>Good to see you, {firstName}.</h1>
            <p>Plan your week, control recurring meals, and see every provider handoff from one private dashboard.</p></div>
          <Link className="ca-primary-action" href="/#discover"><CalendarDays size={17} /> Plan a meal</Link>
        </section>

        {error && <div className="ca-alert error" role="alert"><AlertCircle size={17} />{error}<button onClick={() => load(data.week.starts_on)}>Try again</button></div>}
        {notice && <div className="ca-alert success" role="status"><Check size={17} />{notice}</div>}

        {loading ? <div className="ca-loading"><Loader2 size={24} /> Loading your meal space…</div> : <>
          <section className="ca-next-meal" aria-labelledby="next-meal-title">
            <div className="ca-next-label"><Clock3 size={15} /><span>UPCOMING CONFIRMED MEAL</span></div>
            <div className="ca-next-empty">
              <span className="ca-empty-icon"><CalendarDays size={24} /></span>
              {nextMeal ? <div><h2 id="next-meal-title">{nextMeal.title}</h2><p>{localDateTime(nextMeal.desired_for)} · {pretty(nextMeal.state)}</p></div>
                : <div><h2 id="next-meal-title">Your next meal is open</h2><p>Confirmed scheduled meals will appear here.</p></div>}
              <a href="#planner">View week <ArrowRight size={15} /></a>
            </div>
          </section>

          <section className="ca-planner" id="planner" aria-labelledby="week-title">
            <div className="ca-section-head">
              <div><span>WEEKLY PLANNER</span><h2 id="week-title">{data.week.starts_on && `${localDate(data.week.starts_on)} – ${localDate(data.week.ends_on)}`}</h2></div>
              <div className="ca-week-controls">
                <button aria-label="Previous week" onClick={() => load(shiftWeek(data.week.starts_on, -7))}><ArrowLeft size={16} /></button>
                <button aria-label="Next week" onClick={() => load(shiftWeek(data.week.starts_on, 7))}><ArrowRight size={16} /></button>
              </div>
            </div>
            <div className="ca-week">
              {data.week.days.map(day => <button key={day.date} type="button" className={selectedDate === day.date ? 'active' : ''} aria-pressed={selectedDate === day.date} onClick={() => setSelectedDate(day.date)}>
                <small>{new Date(`${day.date}T12:00:00`).toLocaleDateString('en-IN', { weekday: 'short' })}</small>
                <strong>{day.date.slice(-2)}</strong><i>{day.meals.length || ''}</i>
              </button>)}
            </div>
            <div className="ca-day-plan">
              <div className="ca-day-heading"><div><small>SELECTED DAY</small><h3>{selectedDate && localDate(selectedDate)}</h3></div><span>{selectedDay.meals.length} meals planned</span></div>
              <div className="ca-meal-list">
                {selectedDay.meals.length ? selectedDay.meals.map(meal => <article key={meal.id}>
                  <div className="ca-meal-time"><strong>{meal.local_time}</strong><small>{pretty(meal.meal_period)}</small></div>
                  <div><h4>{meal.title}</h4><p>{pretty(meal.state)} · {money(meal.estimated_cost_cents, meal.currency)}</p></div>
                  {meal.can_skip && <div className="ca-meal-actions">
                    <button disabled={busy === `meal-${meal.id}`} onClick={() => run(`meal-${meal.id}`, () => applyCustomerRoutineAction(meal.routine_id, { action: 'SKIP', occurrence_date: meal.local_date, reason: 'Skipped from customer dashboard' }), 'Meal skipped.')}><SkipForward size={14} /> Skip</button>
                    <input aria-label={`New time for ${meal.title}`} type="datetime-local" min={dateTimeInput(new Date())} value={rescheduleTimes[meal.id] || dateTimeInput(meal.desired_for)} onChange={event => setRescheduleTimes({ ...rescheduleTimes, [meal.id]: event.target.value })} />
                    <button disabled={busy === `meal-${meal.id}`} onClick={() => run(`meal-${meal.id}`, () => applyCustomerRoutineAction(meal.routine_id, { action: 'RESCHEDULE', occurrence_date: meal.local_date, replacement_for: new Date(rescheduleTimes[meal.id] || dateTimeInput(meal.desired_for)).toISOString(), reason: 'Rescheduled from customer dashboard' }), 'Meal rescheduled.')}><CalendarClock size={14} /> Move</button>
                  </div>}
                </article>) : <div className="ca-empty-state">No meals planned for this day. <Link href="/#discover">Start planning</Link></div>}
              </div>
            </div>
          </section>

          <section className="ca-dashboard-grid">
            <article className="ca-panel" id="routines">
              <header><span><RotateCcw size={18} /></span><div><small>TEMPLATES & ROUTINES</small><h2>Your repeatable meals</h2></div></header>
              <div className="ca-stack">
                {data.routines.map(routine => <div className="ca-row" key={routine.id}><div><strong>{routine.name}</strong><small>{pretty(routine.meal_period)} · {routine.recurrence_rule}</small></div>
                  <button disabled={busy === `routine-${routine.id}`} onClick={() => run(`routine-${routine.id}`, () => applyCustomerRoutineAction(routine.id, { action: routine.state === 'ACTIVE' ? 'PAUSE' : 'RESUME', reason: 'Changed from customer dashboard' }), `Routine ${routine.state === 'ACTIVE' ? 'paused' : 'resumed'}.`)}>
                    {routine.state === 'ACTIVE' ? <Pause size={13} /> : <Play size={13} />}{routine.state === 'ACTIVE' ? 'Pause' : 'Resume'}
                  </button></div>)}
                {!data.routines.length && <p className="ca-empty-copy">No recurring routines yet.</p>}
                {!!data.templates.length && <p className="ca-meta">{data.templates.length} saved template{data.templates.length === 1 ? '' : 's'}</p>}
              </div>
            </article>

            <article className="ca-panel">
              <header><span><CalendarDays size={18} /></span><div><small>DRAFT PLANS</small><h2>Review before confirming</h2></div></header>
              <div className="ca-stack">
                {data.draft_plans.map(plan => <div className="ca-row" key={plan.id}><div><strong>{plan.name}</strong><small>{localDate(plan.starts_on)} · {money(plan.estimated_total_cents, plan.currency)}</small></div>
                  <button disabled={busy === `plan-${plan.id}`} onClick={() => run(`plan-${plan.id}`, () => confirmCustomerMealPlan(plan.id), 'Meal plan confirmed. No order was placed.')}>Confirm</button></div>)}
                {!data.draft_plans.length && <p className="ca-empty-copy">No draft plans awaiting review.</p>}
              </div>
            </article>

            <form className="ca-panel ca-form" id="preferences" onSubmit={savePreferences}>
              <header><span><Settings2 size={18} /></span><div><small>MEAL PREFERENCES</small><h2>Diet, allergies & goals</h2></div></header>
              <label>Dietary choices<input value={preferenceForm.dietary_flags} onChange={event => setPreferenceForm({ ...preferenceForm, dietary_flags: event.target.value })} placeholder="vegetarian, high-protein" /></label>
              <label>Allergies<input value={preferenceForm.allergies} onChange={event => setPreferenceForm({ ...preferenceForm, allergies: event.target.value })} placeholder="peanuts, shellfish" /></label>
              <label>Health goals<input value={preferenceForm.health_goals} onChange={event => setPreferenceForm({ ...preferenceForm, health_goals: event.target.value })} placeholder="balanced, lower sodium" /></label>
              <label>Price preference<select value={preferenceForm.price_band} onChange={event => setPreferenceForm({ ...preferenceForm, price_band: event.target.value })}><option value="budget">Budget</option><option value="mid">Mid-range</option><option value="premium">Premium</option></select></label>
              <button className="ca-save" disabled={busy === 'preferences'}>{busy === 'preferences' ? 'Saving…' : 'Save preferences'}</button>
            </form>

            <form className="ca-panel ca-form" onSubmit={saveBudget}>
              <header><span><WalletCards size={18} /></span><div><small>BUDGET & SPEND</small><h2>{money(data.spending.week_cents)} this week</h2></div></header>
              <div className="ca-budget-track"><i style={{ width: `${budgetPercent}%` }} /></div>
              <p className="ca-meta">{activeBudget ? `${budgetPercent}% of ${money(activeBudget.limit_cents, activeBudget.currency)} ${pretty(activeBudget.period)} limit` : 'No active budget limit'} · {money(data.spending.lifetime_cents)} lifetime</p>
              <label>Period<select value={budgetForm.period} onChange={event => setBudgetForm({ ...budgetForm, period: event.target.value })}><option value="WEEKLY">Weekly</option><option value="MONTHLY">Monthly</option></select></label>
              <label>Limit in ₹<input required min="1" step="1" type="number" value={budgetForm.amount} onChange={event => setBudgetForm({ ...budgetForm, amount: event.target.value })} /></label>
              <label>Starts on<input required type="date" value={budgetForm.starts_on} onChange={event => setBudgetForm({ ...budgetForm, starts_on: event.target.value })} /></label>
              <button className="ca-save" disabled={busy === 'budget'}>{busy === 'budget' ? 'Saving…' : 'Update budget'}</button>
              <small className="ca-ledger-note"><ShieldCheck size={13} /> Totals reconcile to the immutable spending ledger.</small>
            </form>

            <article className="ca-panel ca-wide">
              <header><span><ShieldCheck size={18} /></span><div><small>PRIVACY CONTROLS</small><h2>You choose what NourishBot may use</h2></div></header>
              <div className="ca-consent-grid">{Object.entries(data.consents).map(([purpose, granted]) => <label key={purpose}><span><strong>{pretty(purpose)}</strong><small>{granted ? 'Allowed' : 'Not allowed'}</small></span><input type="checkbox" checked={granted} disabled={busy === `consent-${purpose}`} onChange={event => run(`consent-${purpose}`, () => updateCustomerConsent(purpose, event.target.checked), `${pretty(purpose)} consent updated.`)} /></label>)}</div>
              <div className="ca-provider-strip">{data.provider_capabilities.map(provider => <span key={provider.provider}><strong>{provider.provider}</strong><small>{provider.mode} · {provider.ordering_available ? 'Ordering available' : 'Discovery only'}</small></span>)}</div>
            </article>

            <article className="ca-panel ca-wide" id="activity">
              <header><span><ShoppingBag size={18} /></span><div><small>ORDER & PROVIDER HISTORY</small><h2>Every fulfillment handoff</h2></div><button className="ca-refresh" onClick={() => load(data.week.starts_on)}><RefreshCw size={14} /> Refresh</button></header>
              <div className="ca-order-table">
                {data.orders.map(order => <div className="ca-order-row" key={order.id}>
                  <div><strong>{order.order_ref}</strong><small>{order.provider} · {pretty(order.order_type)}</small></div>
                  <div><strong>{pretty(order.status)}</strong><small>Submitted {localDateTime(order.submitted_at)} · Completed {localDateTime(order.completed_at)}</small></div>
                  <div><strong>{money(order.total_cents, order.currency)}</strong><small>{attemptsByOrder[order.id]?.[0] ? `${attemptsByOrder[order.id][0].provider}: ${pretty(attemptsByOrder[order.id][0].status)}` : 'No external handoff'}</small></div>
                  {order.can_cancel && <button disabled={busy === `order-${order.id}`} onClick={() => run(`order-${order.id}`, () => cancelCustomerOrder(order.id), 'Order cancellation confirmed.')}>Cancel</button>}
                </div>)}
                {!data.orders.length && <p className="ca-empty-copy">No order history yet.</p>}
              </div>
            </article>
          </section>
        </>}
      </div>
    </main>
  )
}
