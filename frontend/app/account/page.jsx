'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../context/AuthContext'
import {
  ArrowRight,
  CalendarDays,
  Clock3,
  Heart,
  IndianRupee,
  Leaf,
  LogOut,
  MessageCircle,
  Plus,
  Settings2,
  ShoppingBag,
  UtensilsCrossed,
} from 'lucide-react'

const days = [
  { day: 'Mon', date: '19' },
  { day: 'Tue', date: '20' },
  { day: 'Wed', date: '21' },
  { day: 'Thu', date: '22' },
  { day: 'Fri', date: '23' },
  { day: 'Sat', date: '24' },
  { day: 'Sun', date: '25' },
]

const mealSlots = [
  { label: 'Breakfast', time: '7–10 AM' },
  { label: 'Lunch', time: '12–3 PM' },
  { label: 'Snacks', time: '4–6 PM' },
  { label: 'Dinner', time: '7–10 PM' },
]

export default function AccountPage() {
  const { user, role, isLoggedIn, authLoading, logout } = useAuth()
  const router = useRouter()
  const [activeDate, setActiveDate] = useState('21')

  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.replace('/login')
    else if (!authLoading && role && role !== 'CUSTOMER') router.replace('/workspace')
  }, [authLoading, isLoggedIn, role, router])

  if (authLoading || !isLoggedIn || role !== 'CUSTOMER') return null

  async function signOut() {
    await logout()
    router.replace('/login')
  }

  const firstName = user.name ? user.name.split(' ')[0] : 'there'

  return (
    <main className="customer-account-page ca-page">
      <header className="ca-nav">
        <Link className="ca-brand" href="/" aria-label="NourishBot home">
          <span><Leaf size={18} /></span>NourishBot
        </Link>
        <div className="ca-nav-links" aria-label="Customer navigation">
          <a className="active" href="#planner">Planner</a>
          <a href="#preferences">Preferences</a>
          <a href="#activity">Activity</a>
        </div>
        <div className="ca-nav-actions">
          <span className="ca-user-dot">{firstName.charAt(0).toUpperCase()}</span>
          <button type="button" onClick={signOut}><LogOut size={15} /> <span>Sign out</span></button>
        </div>
      </header>

      <div className="ca-shell">
        <section className="ca-welcome">
          <div>
            <span className="ca-eyebrow">YOUR MEAL SPACE</span>
            <h1>Good to see you, {firstName}.</h1>
            <p>Plan the week one meal at a time. Nothing moves forward until you review and confirm it.</p>
          </div>
          <Link className="ca-primary-action" href="/#discover"><Plus size={17} /> Plan a meal</Link>
        </section>

        <section className="ca-next-meal" aria-labelledby="next-meal-title">
          <div className="ca-next-label"><Clock3 size={15} /><span>UPCOMING MEAL</span></div>
          <div className="ca-next-empty">
            <span className="ca-empty-icon"><CalendarDays size={24} /></span>
            <div><h2 id="next-meal-title">Your next meal is open</h2><p>Start a plan and a verified schedule will appear here.</p></div>
            <Link href="/#discover">Start planning <ArrowRight size={15} /></Link>
          </div>
        </section>

        <section className="ca-planner" id="planner" aria-labelledby="week-title">
          <div className="ca-section-head">
            <div><span>WEEKLY PLANNER</span><h2 id="week-title">October 19–25</h2></div>
            <span className="ca-week-status">Current week</span>
          </div>

          <div className="ca-week">
            {days.map((item) => (
              <button key={item.date} type="button" aria-pressed={activeDate === item.date} className={activeDate === item.date ? 'active' : ''} onClick={() => setActiveDate(item.date)}>
                <small>{item.day}</small><strong>{item.date}</strong>
              </button>
            ))}
          </div>

          <div className="ca-day-plan">
            <div className="ca-day-heading"><div><small>SELECTED DAY</small><h3>{days.find((day) => day.date === activeDate)?.day}, October {activeDate}</h3></div><span>0 meals planned</span></div>
            <div className="ca-meal-slots">
              {mealSlots.map((slot) => (
                <article key={slot.label}>
                  <div><span className="ca-meal-dot" /><p><strong>{slot.label}</strong><small>{slot.time}</small></p></div>
                  <Link href="/#discover" aria-label={`Plan ${slot.label.toLowerCase()}`}><Plus size={16} /> Plan</Link>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="ca-lower-grid">
          <article className="ca-preferences" id="preferences">
            <div className="ca-panel-heading"><span><Settings2 size={18} /></span><div><small>SAVED PREFERENCES</small><h2>Make every search feel more like you.</h2></div></div>
            <div className="ca-empty-row"><Leaf size={18} /><div><strong>No preferences saved yet</strong><p>Add dietary choices and budget guidance during your next plan.</p></div></div>
            <Link href="/#discover">Explore preferences <ArrowRight size={15} /></Link>
          </article>

          <article className="ca-budget">
            <div className="ca-panel-heading"><span><IndianRupee size={18} /></span><div><small>WEEKLY BUDGET</small><h2>Not set</h2></div></div>
            <div className="ca-budget-track"><i /></div>
            <p>Add a budget while planning to make relevant options easier to find.</p>
            <Link href="/#discover">Set during planning <ArrowRight size={15} /></Link>
          </article>

          <article className="ca-activity" id="activity">
            <div className="ca-panel-heading"><span><ShoppingBag size={18} /></span><div><small>RECENT ACTIVITY</small><h2>No confirmed activity</h2></div></div>
            <p>Verified schedules, provider handoffs, and completed meals will appear here.</p>
          </article>

          <article className="ca-favorites">
            <div className="ca-panel-heading"><span><Heart size={18} /></span><div><small>FAVORITES</small><h2>Keep the good ones close.</h2></div></div>
            <p>Restaurants and meals you save will be easy to return to.</p>
          </article>
        </section>

        <section className="ca-shortcuts" aria-label="Meal planning shortcuts">
          <div><MessageCircle size={18} /><span><strong>Plan tomorrow’s lunch</strong><small>Start with time and budget</small></span><Link href="/#discover" aria-label="Plan tomorrow's lunch"><ArrowRight size={16} /></Link></div>
          <div><UtensilsCrossed size={18} /><span><strong>Find dinner on my route</strong><small>Compare practical pickup stops</small></span><Link href="/#about" aria-label="Learn about route planning"><ArrowRight size={16} /></Link></div>
        </section>
      </div>
    </main>
  )
}
