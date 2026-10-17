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
  confirmPlannedMealOrder,
  createCustomerMealPlan,
  fetchCustomerMealMenu,
  fetchCustomerDashboard,
  getCustomerMealPlan,
  previewPlannedMealOrder,
  updateCustomerBudget,
  updateCustomerConsent,
  updateCustomerPlannedMeal,
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
  const [nutritionTargets, setNutritionTargets] = useState({
    energy_kcal: '', protein_g: '', carbohydrates_g: '', fat_g: '', fiber_g: '',
  })
  const [nutritionPlan, setNutritionPlan] = useState(null)
  const [planReview, setPlanReview] = useState(null)
  const [planMenuItems, setPlanMenuItems] = useState([])
  const [mealEdits, setMealEdits] = useState({})
  const [budgetForm, setBudgetForm] = useState({ period: 'WEEKLY', amount: '', starts_on: '' })
  const [rescheduleTimes, setRescheduleTimes] = useState({})
  const [checkout, setCheckout] = useState(null)

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

  async function createNutritionPlan(event) {
    event.preventDefault()
    const targets = Object.fromEntries(
      Object.entries(nutritionTargets).map(([key, value]) => [key, Number(value)])
    )
    await run('nutrition-plan', async () => {
      const todayInIndia = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date())
      const result = await createCustomerMealPlan({
        scope: 'DAILY',
        starts_on: shiftWeek(todayInIndia, 1),
        timezone: 'Asia/Kolkata',
        dietary_flags: splitTerms(preferenceForm.dietary_flags),
        allergies: splitTerms(preferenceForm.allergies),
        health_goals: splitTerms(preferenceForm.health_goals),
        nutrition_targets: targets,
      })
      setNutritionPlan(result)
    }, 'Daily nutrition plan created for review. No order was placed.')
  }

  async function openMealPlanReview(planId) {
    setBusy(`review-plan-${planId}`)
    setError('')
    try {
      const [plan, menuItems] = await Promise.all([
        getCustomerMealPlan(planId),
        fetchCustomerMealMenu(),
      ])
      setPlanReview(plan)
      setPlanMenuItems(menuItems)
      setMealEdits(Object.fromEntries((plan.meals || []).map(meal => [
        meal.id,
        { menu_item_id: '', desired_for: dateTimeInput(meal.desired_for) },
      ])))
    } catch (err) {
      setError(err.message || 'The meal plan could not be loaded for review.')
    } finally {
      setBusy('')
    }
  }

  async function savePlannedMealEdit(meal) {
    if (!planReview) return
    const values = mealEdits[meal.id]
    if (!values?.desired_for) {
      setError('Choose a valid meal time before saving.')
      return
    }
    setBusy(`meal-edit-${meal.id}`)
    setError('')
    setNotice('')
    try {
      const payload = { desired_for: new Date(values.desired_for).toISOString() }
      if (values.menu_item_id) payload.menu_item_id = Number(values.menu_item_id)
      await updateCustomerPlannedMeal(planReview.id, meal.id, payload)
      const updatedPlan = await getCustomerMealPlan(planReview.id)
      setPlanReview(updatedPlan)
      setNutritionPlan(updatedPlan)
      setMealEdits(Object.fromEntries((updatedPlan.meals || []).map(updatedMeal => [
        updatedMeal.id,
        { menu_item_id: '', desired_for: dateTimeInput(updatedMeal.desired_for) },
      ])))
      setNotice('Meal updated. The plan returned to draft and must be reviewed again.')
      await load(data.week.starts_on)
    } catch (err) {
      setError(err.message || 'The planned meal could not be updated.')
    } finally {
      setBusy('')
    }
  }

  function openPlannedMealCheckout(meal) {
    setError('')
    setNotice('')
    setCheckout({
      mealId: meal.id,
      title: meal.title,
      orderType: 'DELIVERY',
      deliveryAddress: '',
      contactPhone: '',
      preview: null,
    })
  }

  // Step one: the server rechecks the plan, menu, and price, then stores the exact
  // order this customer may confirm. Nothing is ordered here.
  async function reviewPlannedMealOrder(event) {
    event.preventDefault()
    if (!checkout) return
    setBusy('checkout-review')
    setError('')
    setNotice('')
    try {
      const preview = await previewPlannedMealOrder(checkout.mealId, {
        order_type: checkout.orderType,
        delivery_address: checkout.deliveryAddress,
        contact_phone: checkout.contactPhone,
      })
      setCheckout(current => (current ? { ...current, preview } : current))
    } catch (err) {
      setCheckout(current => (current ? { ...current, preview: null } : current))
      setError(err.message || 'This planned meal could not be prepared for ordering.')
    } finally {
      setBusy('')
    }
  }

  // Step two: the explicit confirmation. The server rejects it if anything in the
  // reviewed summary changed, and the same confirmation can only place one order.
  async function placePlannedMealOrder() {
    if (!checkout?.preview) return
    setBusy('checkout-confirm')
    setError('')
    setNotice('')
    try {
      const result = await confirmPlannedMealOrder(checkout.mealId, {
        confirmation_id: checkout.preview.confirmation_id,
        order_type: checkout.orderType,
        delivery_address: checkout.deliveryAddress,
      })
      setCheckout(null)
      setNotice(`Order ${result.order_ref} confirmed for ${localDateTime(result.scheduled_for)}.`)
      await load(data.week.starts_on)
    } catch (err) {
      setCheckout(current => (current ? { ...current, preview: null } : current))
      setError(err.message || 'The order was not placed. Review the meal again.')
    } finally {
      setBusy('')
    }
  }

  async function copyPlannedMealOrderCommand(meal) {
    const command = `order planned meal #${meal.id}`
    setError('')
    setNotice('')
    if (!navigator.clipboard?.writeText) {
      setError(`Clipboard access is unavailable. Copy "${command}" and send it in your linked WhatsApp or Telegram chat.`)
      return
    }
    try {
      await navigator.clipboard.writeText(command)
      setNotice(`Copied "${command}". Send it in your linked WhatsApp or Telegram chat to review the order. Nothing has been ordered.`)
    } catch {
      setError(`Could not copy the command. Copy "${command}" and send it in your linked WhatsApp or Telegram chat.`)
    }
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
              <div><span>WEEKLY PLANNER</span><h2 id="week-title">{data.week.starts_on && `${localDate(data.week.starts_on)} to ${localDate(data.week.ends_on)}`}</h2></div>
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
                {selectedDay.meals.length ? selectedDay.meals.map(meal => {
                  const canOrderPlannedMeal = (
                    meal.plan_id
                    && meal.state === 'CONFIRMED'
                    && !meal.order_id
                    && new Date(meal.desired_for).getTime() > Date.now()
                  )
                  const mealCheckout = checkout?.mealId === meal.id ? checkout : null
                  return <article key={meal.id}>
                    <div className="ca-meal-time"><strong>{meal.local_time}</strong><small>{pretty(meal.meal_period)}</small></div>
                    <div><h4>{meal.title}</h4><p>{pretty(meal.state)} · {money(meal.estimated_cost_cents, meal.currency)}</p></div>
                    {(meal.can_skip || canOrderPlannedMeal) && <div className="ca-meal-actions">
                      {meal.can_skip && <>
                        <button disabled={busy === `meal-${meal.id}`} onClick={() => run(`meal-${meal.id}`, () => applyCustomerRoutineAction(meal.routine_id, { action: 'SKIP', occurrence_date: meal.local_date, reason: 'Skipped from customer dashboard' }), 'Meal skipped.')}><SkipForward size={14} /> Skip</button>
                        <input aria-label={`New time for ${meal.title}`} type="datetime-local" min={dateTimeInput(new Date())} value={rescheduleTimes[meal.id] || dateTimeInput(meal.desired_for)} onChange={event => setRescheduleTimes({ ...rescheduleTimes, [meal.id]: event.target.value })} />
                        <button disabled={busy === `meal-${meal.id}`} onClick={() => run(`meal-${meal.id}`, () => applyCustomerRoutineAction(meal.routine_id, { action: 'RESCHEDULE', occurrence_date: meal.local_date, replacement_for: new Date(rescheduleTimes[meal.id] || dateTimeInput(meal.desired_for)).toISOString(), reason: 'Rescheduled from customer dashboard' }), 'Meal rescheduled.')}><CalendarClock size={14} /> Move</button>
                      </>}
                      {canOrderPlannedMeal && <>
                        <button type="button" aria-expanded={!!mealCheckout} onClick={() => (mealCheckout ? setCheckout(null) : openPlannedMealCheckout(meal))}>
                          <ShoppingBag size={14} /> {mealCheckout ? 'Close' : 'Order this meal'}
                        </button>
                        <button type="button" onClick={() => copyPlannedMealOrderCommand(meal)}>
                          Copy order command
                        </button>
                      </>}
                    </div>}
                    {mealCheckout && <form className="ca-meal-checkout" onSubmit={reviewPlannedMealOrder} aria-label={`Order ${meal.title}`}>
                      <p className="ca-meta">Ordering only this meal, for its planned time. Your meal plan stays unchanged.</p>
                      <div className="ca-checkout-fields">
                        <label>How to collect it
                          <select value={mealCheckout.orderType} onChange={event => setCheckout({ ...mealCheckout, orderType: event.target.value, preview: null })}>
                            <option value="DELIVERY">Home delivery</option>
                            <option value="PICKUP">Pickup</option>
                          </select>
                        </label>
                        {mealCheckout.orderType === 'DELIVERY' && <label>Delivery address
                          <input required value={mealCheckout.deliveryAddress} onChange={event => setCheckout({ ...mealCheckout, deliveryAddress: event.target.value, preview: null })} />
                        </label>}
                        <label>Contact mobile
                          <input inputMode="numeric" placeholder="Required the first time you order" value={mealCheckout.contactPhone} onChange={event => setCheckout({ ...mealCheckout, contactPhone: event.target.value, preview: null })} />
                        </label>
                      </div>
                      {!mealCheckout.preview && <button type="submit" disabled={busy === 'checkout-review'}>
                        {busy === 'checkout-review' ? 'Checking the menu…' : 'Review this order'}
                      </button>}
                      {mealCheckout.preview && <div className="ca-checkout-review" aria-live="polite">
                        <strong>Confirm this exact order</strong>
                        {(mealCheckout.preview.items || []).map(item => <p key={item.item_id}>
                          {item.qty} × {item.item_name} · {money(item.unit_price_cents * item.qty, mealCheckout.preview.currency)}
                        </p>)}
                        <p>Total {money(mealCheckout.preview.total_cents, mealCheckout.preview.currency)} · {pretty(mealCheckout.preview.order_type)}</p>
                        <p>{mealCheckout.preview.scheduled_local_text}</p>
                        {mealCheckout.preview.delivery_address && <p>{mealCheckout.preview.delivery_address}</p>}
                        {mealCheckout.preview.price_changed_since_plan && <p className="ca-meta">
                          The current price differs from the plan estimate of {money(mealCheckout.preview.plan_estimate_cents, mealCheckout.preview.currency)}.
                        </p>}
                        <p className="ca-meta">{mealCheckout.preview.message}</p>
                        <div className="ca-meal-actions">
                          <button type="button" disabled={busy === 'checkout-confirm'} onClick={placePlannedMealOrder}>
                            <Check size={14} /> {busy === 'checkout-confirm' ? 'Placing…' : 'Confirm and place order'}
                          </button>
                          <button type="button" onClick={() => setCheckout({ ...mealCheckout, preview: null })}>Change details</button>
                        </div>
                      </div>}
                    </form>}
                  </article>
                }) : <div className="ca-empty-state">No meals planned for this day. <Link href="/#discover">Start planning</Link></div>}
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
                {data.draft_plans.map(plan => <div className="ca-row" key={plan.id}>
                  <div><strong>{plan.name}</strong><small>{localDate(plan.starts_on)} · {money(plan.estimated_total_cents, plan.currency)}</small></div>
                  <button disabled={busy === `review-plan-${plan.id}`} onClick={() => openMealPlanReview(plan.id)}>
                    {busy === `review-plan-${plan.id}` ? 'Loading…' : 'Review / edit'}
                  </button>
                  <button disabled={busy === `plan-${plan.id}`} onClick={() => run(`plan-${plan.id}`, () => confirmCustomerMealPlan(plan.id), 'Meal plan confirmed. No order was placed.')}>Confirm plan</button>
                </div>)}
                {!data.draft_plans.length && <p className="ca-empty-copy">No draft plans awaiting review.</p>}
                {planReview && <section className="ca-stack" aria-label="Meal plan review">
                  <div className="ca-row">
                    <strong>Review {planReview.name}</strong>
                    <button type="button" onClick={() => setPlanReview(null)}>Close</button>
                  </div>
                  <p className="ca-meta">Changing any meal invalidates the previous plan confirmation. Save each meal, then review the revised plan. Plan confirmation does not place an order.</p>
                  {(planReview.meals || []).map(meal => {
                    const values = mealEdits[meal.id] || {}
                    return <div className="ca-row ca-plan-meal-edit" key={meal.id}>
                      <div>
                        <strong>{pretty(meal.meal_period)}: {meal.title}</strong>
                        <small>{money(meal.estimated_cost_cents, meal.currency)} · {pretty(meal.nutrition?.status)} nutrition</small>
                      </div>
                      <label>Change menu item
                        <select
                          value={values.menu_item_id || ''}
                          onChange={event => setMealEdits(current => ({
                            ...current,
                            [meal.id]: { ...values, menu_item_id: event.target.value },
                          }))}
                        >
                          <option value="">Keep current: {meal.title}</option>
                          {planMenuItems.map(item => <option key={item.id} value={item.id}>
                            {item.name} · {money(item.price_cents, planReview.currency)}
                          </option>)}
                        </select>
                      </label>
                      <label>Meal time
                        <input
                          type="datetime-local"
                          required
                          value={values.desired_for || ''}
                          onChange={event => setMealEdits(current => ({
                            ...current,
                            [meal.id]: { ...values, desired_for: event.target.value },
                          }))}
                        />
                      </label>
                      <button
                        type="button"
                        disabled={busy === `meal-edit-${meal.id}`}
                        onClick={() => savePlannedMealEdit(meal)}
                      >
                        {busy === `meal-edit-${meal.id}` ? 'Saving…' : 'Save meal'}
                      </button>
                    </div>
                  })}
                </section>}
              </div>
            </article>

            <form className="ca-panel ca-form" onSubmit={createNutritionPlan}>
              <header><span><ShieldCheck size={18} /></span><div><small>DAILY NUTRITION PLAN</small><h2>Plan meals for one day</h2></div></header>
              <p className="ca-meta">Enter daily targets from a qualified professional. NourishBot does not calculate medical targets. A plan is a draft and does not place orders.</p>
              <label>Energy (kcal)<input type="number" min="1" step="1" value={nutritionTargets.energy_kcal} onChange={event => setNutritionTargets({ ...nutritionTargets, energy_kcal: event.target.value })} required /></label>
              <label>Protein (g)<input type="number" min="0.1" step="0.1" value={nutritionTargets.protein_g} onChange={event => setNutritionTargets({ ...nutritionTargets, protein_g: event.target.value })} required /></label>
              <label>Carbohydrates (g)<input type="number" min="0.1" step="0.1" value={nutritionTargets.carbohydrates_g} onChange={event => setNutritionTargets({ ...nutritionTargets, carbohydrates_g: event.target.value })} required /></label>
              <label>Fat (g)<input type="number" min="0.1" step="0.1" value={nutritionTargets.fat_g} onChange={event => setNutritionTargets({ ...nutritionTargets, fat_g: event.target.value })} required /></label>
              <label>Fibre (g)<input type="number" min="0.1" step="0.1" value={nutritionTargets.fiber_g} onChange={event => setNutritionTargets({ ...nutritionTargets, fiber_g: event.target.value })} required /></label>
              <button className="ca-save" disabled={busy === 'nutrition-plan'}>{busy === 'nutrition-plan' ? 'Planning…' : 'Create daily draft'}</button>
              {nutritionPlan && <section className="ca-stack" aria-label="Generated nutrition plan">
                <strong>{nutritionPlan.name}</strong>
                {nutritionPlan.daily_nutrition?.map(day => <div key={day.date}>
                  <small>{day.status === 'VERIFIED' ? 'Verified menu nutrition totals' : 'Nutrition data incomplete'} · {day.date}</small>
                  <p className="ca-meta">
                    {Math.round(day.totals.energy_kcal)} kcal · {day.totals.protein_g.toFixed(1)} g protein · {day.totals.carbohydrates_g.toFixed(1)} g carbs · {day.totals.fat_g.toFixed(1)} g fat · {day.totals.fiber_g.toFixed(1)} g fibre
                  </p>
                  {day.difference_from_targets && <small>
                    Difference from targets (menu total − target): {Math.round(day.difference_from_targets.energy_kcal)} kcal · {day.difference_from_targets.protein_g.toFixed(1)} g protein · {day.difference_from_targets.carbohydrates_g.toFixed(1)} g carbs · {day.difference_from_targets.fat_g.toFixed(1)} g fat · {day.difference_from_targets.fiber_g.toFixed(1)} g fibre
                  </small>}
                  {day.status !== 'VERIFIED' && <small>Partial totals include verified meal values only. Target differences are withheld while any meal data is incomplete.</small>}
                </div>)}
                {nutritionPlan.meals?.map(meal => <div className="ca-row" key={meal.id}>
                  <div><strong>{pretty(meal.meal_period)}: {meal.title}</strong><small>{localDateTime(meal.desired_for)} · {money(meal.estimated_cost_cents, meal.currency)}</small>
                    <small>{pretty(meal.nutrition?.status)} nutrition · {meal.nutrition?.serving_g ? `${meal.nutrition.serving_g} g serving` : 'serving size unavailable'}{meal.nutrition?.source ? ` · Source: ${meal.nutrition.source}` : ''}</small>
                  </div>
                </div>)}
                <button type="button" className="ca-save" onClick={() => openMealPlanReview(nutritionPlan.id)}>
                  Review / adjust meals
                </button>
                <small>Review and confirm each order separately. Confirming this plan alone places no food order.</small>
              </section>}
            </form>

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
