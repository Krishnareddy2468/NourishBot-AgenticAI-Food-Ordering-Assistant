'use client'
/**
 * Customer-facing delivery tracking page.
 * Accessible via /track/:orderRef — no auth required.
 * Shows order status, driver info, GPS trail, and proof of delivery.
 */

import { useState, useEffect } from 'react'

import { Truck, MapPin, CheckCircle2, Clock, User, Phone, Camera, UtensilsCrossed, Loader2 } from 'lucide-react'
import { trackDeliveryByOrder } from '../../services/platformApi'

const STATUS_STEPS = [
  { key: 'CREATED', label: 'Order Placed', icon: Clock },
  { key: 'ACCEPTED', label: 'Confirmed', icon: CheckCircle2 },
  { key: 'PREPARING', label: 'Preparing', icon: UtensilsCrossed },
  { key: 'READY', label: 'Ready', icon: CheckCircle2 },
  { key: 'ASSIGNED', label: 'Driver Assigned', icon: User },
  { key: 'PICKED_UP', label: 'Picked Up', icon: Truck },
  { key: 'EN_ROUTE', label: 'On the Way', icon: MapPin },
  { key: 'DELIVERED', label: 'Delivered', icon: CheckCircle2 },
]

function getStepIndex(status) {
  if (!status) return -1
  const idx = STATUS_STEPS.findIndex(s => s.key === status)
  return idx >= 0 ? idx : -1
}

function getStepIndexForOrder(orderStatus) {
  const mapping = {
    CREATED: 0, ACCEPTED: 1, PREPARING: 2, READY: 3,
    OUT_FOR_DELIVERY: 5,
  }
  return mapping[orderStatus] ?? -1
}

export default function TrackingPage({ orderRef }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const result = await trackDeliveryByOrder(orderRef)
        setData(result)
      } catch (err) {
        setError(err.message || 'Failed to load tracking info')
      } finally {
        setLoading(false)
      }
    }
    if (orderRef) load()
  }, [orderRef])

  // Auto-refresh every 15s
  useEffect(() => {
    const id = setInterval(async () => {
      if (!orderRef) return
      try {
        const result = await trackDeliveryByOrder(orderRef)
        setData(result)
      } catch {
        // Keep the last successful tracking snapshot during background refresh failures.
      }
    }, 15000)
    return () => clearInterval(id)
  }, [orderRef])

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh', background: '#0D0D16',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: '#F0F0FF', fontFamily: 'Inter, sans-serif',
      }}>
        <Loader2 className="spin" size={32} style={{ marginBottom: 12 }} />
        <div>Tracking your order...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{
        minHeight: '100vh', background: '#0D0D16',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: '#F0F0FF', fontFamily: 'Inter, sans-serif',
      }}>
        <div style={{ color: '#EF4444', marginBottom: 8 }}>Unable to track order</div>
        <div style={{ color: '#888', fontSize: 13 }}>{error}</div>
      </div>
    )
  }

  const delivery = data?.delivery
  const orderStatus = data?.order_status
  const currentStep = delivery
    ? getStepIndex(delivery.status)
    : getStepIndexForOrder(orderStatus)

  return (
    <div style={{
      minHeight: '100vh', background: '#0D0D16',
      color: '#F0F0FF', fontFamily: 'Inter, sans-serif',
      padding: '24px 16px', maxWidth: 480, margin: '0 auto',
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
          <UtensilsCrossed size={20} style={{ display: 'inline', marginRight: 8 }} />
          Dzukku Restaurant
        </div>
        <div style={{ fontSize: 13, color: '#888' }}>
          Order #{orderRef}
        </div>
      </div>

      {/* Progress steps */}
      <div style={{ marginBottom: 24 }}>
        {STATUS_STEPS.map((step, idx) => {
          const Icon = step.icon
          const isCompleted = idx <= currentStep
          const isCurrent = idx === currentStep
          const color = isCompleted ? '#1A936F' : '#333'
          const textColor = isCompleted ? '#F0F0FF' : '#555'

          return (
            <div key={step.key} style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '8px 0', position: 'relative',
            }}>
              {/* Connector line */}
              {idx < STATUS_STEPS.length - 1 && (
                <div style={{
                  position: 'absolute', left: 15, top: 32, width: 2,
                  height: 24, background: isCompleted ? '#1A936F' : '#333',
                }} />
              )}
              <div style={{
                width: 32, height: 32, borderRadius: '50%',
                background: isCompleted ? `${color}22` : '#1a1a28',
                border: `2px solid ${color}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}>
                <Icon size={14} style={{ color }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{
                  fontWeight: isCurrent ? 700 : 500,
                  color: textColor,
                  fontSize: isCurrent ? 14 : 13,
                }}>
                  {step.label}
                </div>
                {isCurrent && delivery?.status === 'EN_ROUTE' && delivery?.last_location && (
                  <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
                    <MapPin size={11} style={{ display: 'inline', marginRight: 4 }} />
                    Driver is nearby
                  </div>
                )}
              </div>
              {isCompleted && (
                <CheckCircle2 size={16} style={{ color: '#1A936F' }} />
              )}
            </div>
          )
        })}
      </div>

      {/* Driver info */}
      {delivery?.driver && (
        <div style={{
          background: '#1A1A28', borderRadius: 12, padding: 16,
          border: '1px solid rgba(255,255,255,0.08)', marginBottom: 16,
        }}>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>YOUR DRIVER</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 40, height: 40, borderRadius: '50%',
              background: 'rgba(26,147,111,0.15)', color: '#1A936F',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 700, fontSize: 16,
            }}>
              {delivery.driver.name?.charAt(0) || '?'}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600 }}>{delivery.driver.name}</div>
              {delivery.driver.phone && (
                <div style={{ fontSize: 12, color: '#888' }}>
                  <Phone size={11} style={{ display: 'inline', marginRight: 4 }} />
                  {delivery.driver.phone}
                </div>
              )}
            </div>
            <div style={{ fontSize: 12, color: '#888' }}>
              {delivery.driver.vehicle_type} {delivery.driver.vehicle_no}
            </div>
          </div>
        </div>
      )}

      {/* Proof of delivery */}
      {delivery?.proof_url && (
        <div style={{
          background: '#1A1A28', borderRadius: 12, padding: 16,
          border: '1px solid rgba(255,255,255,0.08)', marginBottom: 16,
        }}>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>PROOF OF DELIVERY</div>
          <a
            href={delivery.proof_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              color: '#1A936F', fontSize: 13, textDecoration: 'none',
            }}
          >
            <Camera size={14} />
            {delivery.proof_type === 'SIGNATURE' ? 'View Signature' : 'View Photo'}
          </a>
        </div>
      )}

      {/* No delivery assigned yet message */}
      {!delivery && (
        <div style={{
          background: '#1A1A28', borderRadius: 12, padding: 16,
          border: '1px solid rgba(255,255,255,0.08)', marginBottom: 16,
          textAlign: 'center', color: '#888', fontSize: 13,
        }}>
          Your order is being prepared. A driver will be assigned once it's ready.
        </div>
      )}

      {/* Auto-refresh notice */}
      <div style={{ textAlign: 'center', color: '#555', fontSize: 11, marginTop: 24 }}>
        Auto-refreshes every 15 seconds
      </div>
    </div>
  )
}
