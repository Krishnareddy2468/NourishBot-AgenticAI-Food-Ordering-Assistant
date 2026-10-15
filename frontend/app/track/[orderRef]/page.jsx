'use client'

import TrackingPage from '../../../components/pages/TrackingPage'

export default function TrackRoute({ params }) {
  return <TrackingPage trackingToken={params.orderRef} />
}
