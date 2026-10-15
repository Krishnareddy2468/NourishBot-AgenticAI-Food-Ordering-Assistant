'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '../../context/AuthContext'
import {
  getGoogleOAuthUrl,
  getOAuthProviders,
  loginApi,
  signupApi,
} from '../../services/platformApi'
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
  UserRound,
  UtensilsCrossed,
} from 'lucide-react'

const mealPeriods = ['Breakfast', 'Lunch', 'Snacks', 'Dinner']

function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M21.6 12.2c0-.7-.1-1.4-.2-2H12v3.9h5.4a4.6 4.6 0 0 1-2 3v2.5h3.3c1.9-1.8 2.9-4.4 2.9-7.4Z" />
      <path fill="#34A853" d="M12 22c2.7 0 5-.9 6.7-2.4l-3.3-2.5c-.9.6-2.1 1-3.4 1a5.9 5.9 0 0 1-5.5-4.1H3.1v2.6A10 10 0 0 0 12 22Z" />
      <path fill="#FBBC05" d="M6.5 14a6 6 0 0 1 0-3.9V7.4H3.1a10 10 0 0 0 0 9.2L6.5 14Z" />
      <path fill="#EA4335" d="M12 6a5.4 5.4 0 0 1 3.8 1.5l2.9-2.8A9.7 9.7 0 0 0 12 2a10 10 0 0 0-8.9 5.4l3.4 2.7A5.9 5.9 0 0 1 12 6Z" />
    </svg>
  )
}

export default function LoginPage() {
  const { login } = useAuth()
  const [mode, setMode] = useState('signin')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [googleEnabled, setGoogleEnabled] = useState(null)

  useEffect(() => {
    getOAuthProviders()
      .then(providers => setGoogleEnabled(Boolean(providers.google)))
      .catch(() => setGoogleEnabled(false))

    const params = new URLSearchParams(window.location.search)
    const oauthError = params.get('oauth_error')
    if (oauthError) {
      setError(oauthError)
      window.history.replaceState({}, '', window.location.pathname)
    }
  }, [])

  function selectMode(nextMode) {
    setMode(nextMode)
    setError('')
    setPassword('')
    setConfirmPassword('')
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    if (mode === 'signup' && password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setLoading(true)
    try {
      const result = mode === 'signup'
        ? await signupApi(name, email, password)
        : await loginApi(email, password)
      login(result)
    } catch (err) {
      setError(err.message || (mode === 'signup' ? 'Signup failed' : 'Sign in failed'))
    } finally {
      setLoading(false)
    }
  }

  function continueWithGoogle() {
    if (googleEnabled) window.location.assign(getGoogleOAuthUrl())
  }

  const isSignup = mode === 'signup'

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
            <div className="staff-auth-eyebrow">Meal planning, remembered</div>
            <h1>One account.<br /><span>Every meal in sync.</span></h1>
            <p>
              Save your NourishBot account for a simpler path from meal planning to a verified
              pickup, drive-through, or delivery schedule. Staff can use their assigned account too.
            </p>
          </div>

          <div className="staff-auth-meals" aria-label="Supported meal periods">
            {mealPeriods.map((meal, index) => (
              <span key={meal}><i>{String(index + 1).padStart(2, '0')}</i>{meal}</span>
            ))}
          </div>

          <div className="staff-auth-flow">
            <div><span><Check size={13} /></span><p><strong>Your account, your schedule</strong><small>Return to one protected meal-planning profile</small></p></div>
            <div><span><ChefHat size={13} /></span><p><strong>Operations stay separate</strong><small>Public signup never grants staff permissions</small></p></div>
            <div><span><ShoppingBag size={13} /></span><p><strong>Secure handoff</strong><small>HttpOnly sessions keep tokens out of page scripts</small></p></div>
          </div>
        </aside>

        <div className="staff-auth-panel">
          <div className="staff-auth-tabs" role="tablist" aria-label="Account access">
            <button type="button" role="tab" aria-selected={!isSignup} className={!isSignup ? 'active' : ''} onClick={() => selectMode('signin')}>Sign in</button>
            <button type="button" role="tab" aria-selected={isSignup} className={isSignup ? 'active' : ''} onClick={() => selectMode('signup')}>Create account</button>
          </div>

          <div className="staff-auth-panel-copy">
            <span className="staff-auth-secure"><ShieldCheck size={14} /> Protected account access</span>
            <h2>{isSignup ? 'Create your account' : 'Welcome back'}</h2>
            <p>{isSignup ? 'Start a customer account with email or Google.' : 'Sign in to continue to your NourishBot account.'}</p>
          </div>

          <button
            type="button"
            className="staff-auth-google"
            onClick={continueWithGoogle}
            disabled={googleEnabled !== true || loading}
          >
            {googleEnabled === null ? <Loader2 size={18} className="spin" /> : <GoogleMark />}
            {googleEnabled === false ? 'Google sign-in needs setup' : 'Continue with Google'}
          </button>

          <div className="staff-auth-divider"><span>or continue with email</span></div>

          <form onSubmit={handleSubmit} className="staff-auth-form">
            {error && (
              <div className="staff-auth-error" role="alert">
                <AlertTriangle size={15} />
                <span>{error}</span>
              </div>
            )}

            {isSignup && (
              <div className="staff-auth-field">
                <label htmlFor="account-name">Full name</label>
                <div className="staff-auth-input-wrap">
                  <UserRound size={17} />
                  <input id="account-name" type="text" value={name} onChange={event => setName(event.target.value)} placeholder="Your name" autoComplete="name" minLength={2} maxLength={80} required autoFocus />
                </div>
              </div>
            )}

            <div className="staff-auth-field">
              <label htmlFor="account-email">Email</label>
              <div className="staff-auth-input-wrap">
                <Mail size={17} />
                <input id="account-email" type="email" value={email} onChange={event => setEmail(event.target.value)} placeholder="you@example.com" autoComplete="email" required autoFocus={!isSignup} />
              </div>
            </div>

            <div className="staff-auth-field">
              <label htmlFor="account-password">Password</label>
              <div className="staff-auth-input-wrap">
                <LockKeyhole size={17} />
                <input id="account-password" type={showPassword ? 'text' : 'password'} value={password} onChange={event => setPassword(event.target.value)} placeholder={isSignup ? 'At least 10 characters' : 'Enter your password'} autoComplete={isSignup ? 'new-password' : 'current-password'} minLength={isSignup ? 10 : 1} required />
                <button type="button" className="staff-auth-password-toggle" onClick={() => setShowPassword(value => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'}>
                  {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </div>

            {isSignup && (
              <div className="staff-auth-field">
                <label htmlFor="account-password-confirm">Confirm password</label>
                <div className="staff-auth-input-wrap">
                  <LockKeyhole size={17} />
                  <input id="account-password-confirm" type={showPassword ? 'text' : 'password'} value={confirmPassword} onChange={event => setConfirmPassword(event.target.value)} placeholder="Repeat your password" autoComplete="new-password" minLength={10} required />
                </div>
              </div>
            )}

            <button type="submit" className="staff-auth-submit" disabled={loading}>
              {loading ? <Loader2 size={17} className="spin" /> : <>{isSignup ? 'Create account' : 'Sign in'} <ArrowRight size={17} /></>}
            </button>
          </form>

          <div className="staff-auth-note">
            <ShieldCheck size={15} />
            <span><strong>Safe access by default</strong>Customer signup cannot open staff workspaces. Staff should use credentials assigned by an administrator.</span>
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
