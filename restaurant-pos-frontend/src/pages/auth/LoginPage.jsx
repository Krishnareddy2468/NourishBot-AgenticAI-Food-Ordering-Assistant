/**
 * Login page — staff email + password, stores JWT.
 */

import { useState } from 'react'
import { useAuth } from '../../context/AuthContext'
import { loginApi } from '../../services/platformApi'
import { UtensilsCrossed, AlertTriangle, Loader2, ShieldCheck, ChefHat, ConciergeBell } from 'lucide-react'

export default function LoginPage() {
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await loginApi(email, password)
      login(res.access_token, {
        id: res.user_id,
        role: res.role,
        restaurant_id: res.restaurant_id,
        email,
      })
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-hero">
          <div className="login-logo">
            <UtensilsCrossed size={40} />
            <h1>Dzukku POS</h1>
            <p>Restaurant operations, floor service, and kitchen execution in one workspace.</p>
          </div>
          <div className="login-role-preview">
            <div><ShieldCheck size={15} /> Admin control</div>
            <div><ConciergeBell size={15} /> Waiter workflow</div>
            <div><ChefHat size={15} /> Kitchen realtime</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error">
              <AlertTriangle size={14} />
              {error}
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Email</label>
            <input
              type="email"
              className="form-input"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="staff@dzukku.com"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter password"
              required
            />
          </div>

          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? <Loader2 size={16} className="spin" /> : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          Demo: admin@dzukku.com / admin123 or waiter@dzukku.com / waiter123
        </div>
      </div>
    </div>
  )
}
