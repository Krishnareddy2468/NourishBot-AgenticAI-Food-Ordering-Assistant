'use client'
/** Customer-facing pickup tracking page. Accessible without staff authentication. */

import { useEffect, useState } from 'react'
import { CheckCircle2, Clock, ShoppingBag, UtensilsCrossed, Loader2 } from 'lucide-react'
import { trackDeliveryByOrder } from '../../services/platformApi'

const STATUS_STEPS = [
  { key: 'CREATED', label: 'Order placed', icon: Clock },
  { key: 'ACCEPTED', label: 'Canteen confirmed', icon: CheckCircle2 },
  { key: 'PREPARING', label: 'Preparing', icon: UtensilsCrossed },
  { key: 'READY', label: 'Ready for pickup', icon: ShoppingBag },
  { key: 'DELIVERED', label: 'Collected', icon: CheckCircle2 },
]

function getStepIndex(status) {
  const normalized = status === 'PICKED_UP' ? 'DELIVERED' : status
  const index = STATUS_STEPS.findIndex(step => step.key === normalized)
  return index >= 0 ? index : -1
}

export default function TrackingPage({ orderRef }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        setData(await trackDeliveryByOrder(orderRef))
      } catch (err) {
        setError(err.message || 'Failed to load pickup status')
      } finally {
        setLoading(false)
      }
    }
    if (orderRef) load()
  }, [orderRef])

  useEffect(() => {
    const id = setInterval(async () => {
      if (!orderRef) return
      try {
        setData(await trackDeliveryByOrder(orderRef))
      } catch {
        // Preserve the most recent verified snapshot during a refresh failure.
      }
    }, 15000)
    return () => clearInterval(id)
  }, [orderRef])

  if (loading) {
    return (
      <main className="tracking-state">
        <Loader2 className="spin" size={32} />
        <p>Checking your pickup status…</p>
      </main>
    )
  }

  if (error) {
    return (
      <main className="tracking-state">
        <strong>Unable to find this order</strong>
        <p>{error}</p>
      </main>
    )
  }

  const legacyDeliveryStatus = data?.delivery?.status
  const currentStep = getStepIndex(data?.order_status || legacyDeliveryStatus)

  return (
    <main className="pickup-tracking">
      <header className="pickup-tracking-header">
        <span className="tracking-logo"><UtensilsCrossed size={20} /> NourishBot</span>
        <h1>Your meal is moving through the canteen</h1>
        <p>Order <strong>#{orderRef}</strong> · Show this reference at collection.</p>
      </header>

      <section className="pickup-progress" aria-label="Order progress">
        {STATUS_STEPS.map((step, index) => {
          const Icon = step.icon
          const completed = index <= currentStep
          const current = index === currentStep
          return (
            <div className={`pickup-step ${completed ? 'completed' : ''} ${current ? 'current' : ''}`} key={step.key}>
              <span className="pickup-step-icon"><Icon size={16} /></span>
              <div>
                <strong>{step.label}</strong>
                {current && <small>This is the latest verified update.</small>}
              </div>
              {completed && <CheckCircle2 className="pickup-step-check" size={18} />}
            </div>
          )
        })}
      </section>

      <aside className="pickup-note">
        <ShoppingBag size={20} />
        <div>
          <strong>Pickup only</strong>
          <p>NourishBot will not request a driver or delivery address for this demo order.</p>
        </div>
      </aside>

      <footer className="tracking-refresh">Status refreshes every 15 seconds.</footer>
    </main>
  )
}
