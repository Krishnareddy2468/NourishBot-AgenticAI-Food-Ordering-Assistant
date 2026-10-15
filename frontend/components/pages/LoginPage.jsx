'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useAuth } from '../../context/AuthContext'
import { loginApi } from '../../services/platformApi'
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChefHat,
  Eye,
  EyeOff,
  Loader2,
  LockKeyhole,
  Mail,
  ShieldCheck,
  ShoppingBag,
  Sparkles,
  UtensilsCrossed,
} from 'lucide-react'

const mealPeriods = ['Breakfast', 'Lunch', 'Snacks', 'Dinner']

export default function LoginPage() {
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await loginApi(email, password)
      login(res)
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="staff-auth-page">
      <header className="staff-auth-nav">
        <Link className="staff-auth-brand" href="/" aria-label="NourishBot home">
          <span><UtensilsCrossed size={19} /></span>
          NourishBot
        </Link>
        <Link className="staff-auth-back" href="/">
          <ArrowLeft size={14} /> Back to website
        </Link>
      </header>

      <section className="staff-auth-shell">
        <aside className="staff-auth-story">
          <div>
            <div className="staff-auth-eyebrow"><Sparkles size={14} /> Staff coordination</div>
            <h1>One service.<br /><span>Every team in sync.</span></h1>
            <p>
              Coordinate scheduled breakfast, lunch, snacks, and dinner from confirmation
              through timed kitchen preparation and verified pickup, drive-through, or delivery handoff.
            </p>
          </div>

          <div className="staff-auth-meals" aria-label="Supported meal periods">
            {mealPeriods.map((meal, index) => (
              <span key={meal}><i>{String(index + 1).padStart(2, '0')}</i>{meal}</span>
            ))}
          </div>

          <div className="staff-auth-flow">
            <div><span><Check size={13} /></span><p><strong>Order confirmed</strong><small>Validated before it reaches staff</small></p></div>
            <div><span><ChefHat size={13} /></span><p><strong>Kitchen updated</strong><small>Every line item stays visible</small></p></div>
            <div><span><ShoppingBag size={13} /></span><p><strong>Fulfillment verified</strong><small>Pickup, drive-through, or delivery closes the journey</small></p></div>
          </div>
        </aside>

        <div className="staff-auth-panel">
          <div className="staff-auth-panel-copy">
            <span className="staff-auth-secure"><ShieldCheck size={14} /> Authorized staff only</span>
            <h2>Welcome back</h2>
            <p>Sign in to open your NourishBot operations workspace.</p>
          </div>

          <form onSubmit={handleSubmit} className="staff-auth-form">
            {error && (
              <div className="staff-auth-error" role="alert">
                <AlertTriangle size={15} />
                <span>{error}</span>
              </div>
            )}

            <div className="staff-auth-field">
              <label htmlFor="staff-email">Work email</label>
              <div className="staff-auth-input-wrap">
                <Mail size={17} />
                <input
                  id="staff-email"
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="staff@nourishbot.com"
                  autoComplete="email"
                  required
                  autoFocus
                />
              </div>
            </div>

            <div className="staff-auth-field">
              <label htmlFor="staff-password">Password</label>
              <div className="staff-auth-input-wrap">
                <LockKeyhole size={17} />
                <input
                  id="staff-password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  className="staff-auth-password-toggle"
                  onClick={() => setShowPassword(value => !value)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </div>

            <button type="submit" className="staff-auth-submit" disabled={loading}>
              {loading ? <Loader2 size={17} className="spin" /> : <>Continue to workspace <ArrowRight size={17} /></>}
            </button>
          </form>

          <div className="staff-auth-note">
            <ShieldCheck size={15} />
            <span><strong>Portfolio environment</strong>Use demo staff accounts only. Never enter production credentials.</span>
          </div>
        </div>
      </section>

      <footer className="staff-auth-footer">
        <span>Scheduled meals · pickup, drive-through, and delivery</span>
        <span>Breakfast · Lunch · Snacks · Dinner</span>
      </footer>
    </main>
  )
}
