const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

function buildHeaders(extra = {}) {
  return {
    'Content-Type': 'application/json',
    ...extra,
  }
}

function makeIdempotencyKey(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function mapBackendStateToUi(orderState) {
  const mapping = {
    CREATED: 'Pending',
    AWAITING_PAYMENT: 'Pending',
    PAID: 'Pending',
    CONFIRMED_BY_RESTAURANT: 'Accepted',
    PREPARING: 'Preparing',
    READY: 'Ready',
    OUT_FOR_DELIVERY: 'Ready',
    COMPLETED: 'Delivered',
    CANCELLED: 'Cancelled',
    REFUND_PENDING: 'Cancelled',
    REFUNDED: 'Cancelled',
  }
  return mapping[orderState] || orderState
}

export function mapUiStateToBackend(status) {
  const mapping = {
    Pending: 'PAID',
    Accepted: 'CONFIRMED_BY_RESTAURANT',
    Preparing: 'PREPARING',
    Ready: 'READY',
    Delivered: 'COMPLETED',
    Cancelled: 'CANCELLED',
  }
  return mapping[status] || status
}

export function normalizeBackendOrder(order) {
  const items = order.items || []
  const qty = items.reduce((sum, item) => sum + Number(item.qty || 1), 0)
  return {
    id: order.order_id,
    backendState: order.order_state,
    customer: order.customer || 'Guest',
    phone: order.phone || '',
    item: items.map(item => `${item.qty}x ${item.name}`).join(', '),
    items: items.map(item => ({
      ...item,
      finalPrice: Number(item.unit_price ?? item.price ?? 0),
      emoji: item.emoji || '🍽️',
    })),
    qty,
    price: Number(order.price_breakdown?.grand_total || order.amount || 0),
    status: mapBackendStateToUi(order.order_state),
    paymentState: order.payment_state || 'PENDING',
    eta: '20 mins',
    dateTime: order.created_at
      ? new Date(order.created_at).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
      : '',
    platform: order.fulfillment_metadata?.platform || order.platform || 'POS',
    address: order.fulfillment_metadata?.address || '',
    table_no: order.fulfillment_metadata?.table_no || '',
    special: order.fulfillment_metadata?.special_instructions || '',
    timeline: order.status_timeline || [],
    settlement: order.settlement || null,
  }
}

async function parseResponse(res) {
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const error = await res.json()
      message = error.detail || error.message || message
    } catch {
      // ignore json parsing errors on failures
    }
    throw new Error(message)
  }
  return res.json()
}

export async function fetchBackendOrders() {
  const res = await fetch(`${API_BASE}/v1/orders?limit=200`)
  const data = await parseResponse(res)
  return data.map(normalizeBackendOrder)
}

export async function createBackendOrder(payload) {
  const res = await fetch(`${API_BASE}/v1/orders`, {
    method: 'POST',
    headers: buildHeaders({ 'Idempotency-Key': makeIdempotencyKey('order') }),
    body: JSON.stringify(payload),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function createBackendPaymentIntent(payload) {
  const res = await fetch(`${API_BASE}/v1/payments/intents`, {
    method: 'POST',
    headers: buildHeaders({ 'Idempotency-Key': makeIdempotencyKey('payment') }),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

export async function markBackendOrderPaid(orderId) {
  const res = await fetch(`${API_BASE}/v1/orders/${orderId}/mark-paid`, {
    method: 'POST',
    headers: buildHeaders(),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function updateBackendOrderState(orderId, backendState) {
  const res = await fetch(`${API_BASE}/v1/orders/${orderId}/state`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ order_state: backendState }),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function fetchBackendSettlement(orderId) {
  const res = await fetch(`${API_BASE}/v1/settlements/orders/${orderId}`)
  const data = await parseResponse(res)
  return data.fee_breakdown
}
