'use client'
/** Customer-facing status view authorized by a high-entropy tracking token. */

import { useEffect, useState } from 'react'
import { CalendarClock, CheckCircle2, Clock, ShoppingBag, Truck, UtensilsCrossed, Loader2 } from 'lucide-react'
import { trackDeliveryByToken } from '../../services/platformApi'

const COLLECTION_STEPS = [
  { key: 'SCHEDULED', label: 'Meal scheduled', icon: CalendarClock },
  { key: 'CREATED', label: 'Order placed', icon: Clock },
  { key: 'ACCEPTED', label: 'Canteen confirmed', icon: CheckCircle2 },
  { key: 'PREPARING', label: 'Preparing', icon: UtensilsCrossed },
  { key: 'READY', label: 'Ready for pickup', icon: ShoppingBag },
  { key: 'DELIVERED', label: 'Collected', icon: CheckCircle2 },
]

const DELIVERY_STEPS = [
  { key: 'SCHEDULED', label: 'Meal scheduled', icon: CalendarClock },
  { key: 'CREATED', label: 'Order placed', icon: Clock },
  { key: 'ACCEPTED', label: 'Canteen confirmed', icon: CheckCircle2 },
  { key: 'PREPARING', label: 'Preparing', icon: UtensilsCrossed },
  { key: 'READY', label: 'Ready for dispatch', icon: ShoppingBag },
  { key: 'OUT_FOR_DELIVERY', label: 'Out for delivery', icon: Truck },
  { key: 'DELIVERED', label: 'Delivered', icon: CheckCircle2 },
]

function getStepIndex(status, steps) {
  const normalized = status === 'PICKED_UP' ? 'DELIVERED' : status
  const index = steps.findIndex(step => step.key === normalized)
  return index >= 0 ? index : -1
}

export default function TrackingPage({ trackingToken }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        setData(await trackDeliveryByToken(trackingToken))
      } catch (err) {
        setError(err.message || 'Failed to load fulfillment status')
      } finally {
        setLoading(false)
      }
    }
    if (trackingToken) load()
  }, [trackingToken])

  useEffect(() => {
    const id = setInterval(async () => {
      if (!trackingToken) return
      try {
        setData(await trackDeliveryByToken(trackingToken))
      } catch {
        // Preserve the most recent verified snapshot during a refresh failure.
      }
    }, 15000)
    return () => clearInterval(id)
  }, [trackingToken])

  if (loading) {
    return (
      <main className="tracking-state">
        <Loader2 className="spin" size={32} />
        <p>Checking your scheduled meal status…</p>
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
  const isHomeDelivery = data?.order_type === 'DELIVERY'
  const statusSteps = isHomeDelivery ? DELIVERY_STEPS : COLLECTION_STEPS
  const currentStep = getStepIndex(data?.order_status || legacyDeliveryStatus, statusSteps)
  const scheduledLabel = data?.scheduled_for
    ? new Date(data.scheduled_for).toLocaleString('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short',
        timeZone: 'Asia/Kolkata',
      })
    : null

  return (
    <main className="pickup-tracking">
      <header className="pickup-tracking-header">
        <span className="tracking-logo"><UtensilsCrossed size={20} /> NourishBot</span>
        <h1>{currentStep === 0 ? 'Your meal is scheduled' : isHomeDelivery ? 'Your meal is being prepared for delivery' : 'Your meal is moving through preparation'}</h1>
        <p>
          Order <strong>#{data?.order_ref || 'Scheduled meal'}</strong>
          {scheduledLabel
            ? ` · Scheduled for ${scheduledLabel}`
            : isHomeDelivery
              ? ' · Keep this reference for delivery tracking.'
              : ' · Show this reference at collection.'}
        </p>
      </header>

      <section className="pickup-progress" aria-label="Order progress">
        {statusSteps.map((step, index) => {
          const Icon = step.icon
          const completed = index <= currentStep
          const current = index === currentStep
          const label = step.key === 'READY' && data?.order_type === 'DRIVE_THRU'
            ? 'Ready at drive-through'
            : step.label
          return (
            <div className={`pickup-step ${completed ? 'completed' : ''} ${current ? 'current' : ''}`} key={step.key}>
              <span className="pickup-step-icon"><Icon size={16} /></span>
              <div>
                <strong>{label}</strong>
                {current && <small>This is the latest verified update.</small>}
              </div>
              {completed && <CheckCircle2 className="pickup-step-check" size={18} />}
            </div>
          )
        })}
      </section>

      <aside className="pickup-note">
        {isHomeDelivery ? <Truck size={20} /> : <ShoppingBag size={20} />}
        <div>
          <strong>{data?.provider || 'NourishBot'} fulfillment</strong>
          <p>
            {data?.provider && data.provider !== 'NOURISHBOT'
              ? 'The connected provider controls fulfillment and sends its verified updates.'
              : isHomeDelivery
                ? 'Your meal is prepared near the selected time, assigned to a driver, and tracked through delivery.'
                : data?.order_type === 'DRIVE_THRU'
                ? 'Your in-house meal is prepared near the selected time and handed over at the drive-through lane.'
                : 'Your in-house meal is prepared near the selected time and collected at pickup.'}
          </p>
        </div>
      </aside>

      <footer className="tracking-refresh">Status refreshes every 15 seconds.</footer>
    </main>
  )
}
