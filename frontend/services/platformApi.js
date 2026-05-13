/**
 * Dzukku POS API client — all backend endpoints for vNext.
 */

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

function buildHeaders(extra = {}) {
  const token = localStorage.getItem('dzukku_token')
  const h = { 'Content-Type': 'application/json', ...extra }
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

function makeIdempotencyKey(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function mapBackendStateToUi(orderState) {
  const mapping = {
    CREATED: 'Pending', AWAITING_PAYMENT: 'Pending', PAID: 'Pending',
    CONFIRMED_BY_RESTAURANT: 'Accepted', ACCEPTED: 'Accepted',
    PREPARING: 'Preparing', READY: 'Ready', OUT_FOR_DELIVERY: 'Ready',
    COMPLETED: 'Delivered', DELIVERED: 'Delivered', CANCELLED: 'Cancelled',
    REFUND_PENDING: 'Cancelled', REFUNDED: 'Cancelled',
  }
  return mapping[orderState] || orderState
}

export function mapUiStateToBackend(status) {
  const mapping = {
    Pending: 'PAID', Accepted: 'CONFIRMED_BY_RESTAURANT', Preparing: 'PREPARING',
    Ready: 'READY', Delivered: 'COMPLETED', Cancelled: 'CANCELLED',
  }
  return mapping[status] || status
}

export function normalizeBackendOrder(order) {
  const items = order.items || []
  const qty = items.reduce((sum, item) => sum + Number(item.qty || 1), 0)
  return {
    id: order.order_id || order.id,
    orderRef: order.order_ref,
    backendState: order.order_state || order.status,
    customer: order.customer || 'Guest',
    phone: order.phone || '',
    item: items.map(item => `${item.qty}x ${item.name || item.item_name_snapshot}`).join(', '),
    items: items.map(item => ({
      ...item,
      name: item.name || item.item_name_snapshot,
      finalPrice: Number(item.unit_price ?? item.price ?? ((item.unit_price_cents ?? 0) / 100)),
      emoji: item.emoji || '',
    })),
    qty,
    price: Number(order.price_breakdown?.grand_total || order.amount || order.total_cents / 100 || 0),
    status: mapBackendStateToUi(order.order_state || order.status),
    paymentState: order.payment_state || order.payment_status || 'PENDING',
    orderType: order.order_type || 'DELIVERY',
    eta: '20 mins',
    createdAt: order.created_at || '',
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
      // Ignore non-JSON error responses and preserve the fallback message.
    }
    if (res.status === 401) {
      localStorage.removeItem('dzukku_token')
      localStorage.removeItem('dzukku_user')
      window.dispatchEvent(new Event('dzukku-auth-expired'))
    }
    throw new Error(message)
  }
  return res.json()
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function loginApi(email, password) {
  const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ email, password }),
  })
  return parseResponse(res)
}

// ── Orders ───────────────────────────────────────────────────────────────────

export async function fetchBackendOrders() {
  const res = await fetch(`${API_BASE}/api/v1/orders?limit=200`, { headers: buildHeaders() })
  const data = await parseResponse(res)
  return data.map(normalizeBackendOrder)
}

export async function createBackendOrder(payload) {
  const res = await fetch(`${API_BASE}/api/v1/orders`, {
    method: 'POST',
    headers: buildHeaders({ 'Idempotency-Key': makeIdempotencyKey('order') }),
    body: JSON.stringify(payload),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function updateBackendOrderState(orderId, backendState) {
  const res = await fetch(`${API_BASE}/api/v1/orders/${orderId}/state`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ order_state: backendState }),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function markBackendOrderPaid(orderId) {
  const res = await fetch(`${API_BASE}/api/v1/orders/${orderId}/mark-paid`, {
    method: 'POST',
    headers: buildHeaders(),
  })
  return normalizeBackendOrder(await parseResponse(res))
}

export async function updateOrderItemStatus(orderId, itemId, status) {
  const res = await fetch(`${API_BASE}/api/v1/orders/${orderId}/items/${itemId}/status`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ status }),
  })
  return parseResponse(res)
}

// ── Payments ─────────────────────────────────────────────────────────────────

export async function createBackendPaymentIntent(payload) {
  const res = await fetch(`${API_BASE}/api/v1/payments/intents`, {
    method: 'POST',
    headers: buildHeaders({ 'Idempotency-Key': makeIdempotencyKey('payment') }),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

// ── Menu ─────────────────────────────────────────────────────────────────────

export async function fetchMenu() {
  const res = await fetch(`${API_BASE}/api/v1/menu/items`, { headers: buildHeaders() })
  const data = await parseResponse(res)
  // Backend returns { items: [...] }
  return Array.isArray(data) ? data : (data.items || [])
}

export async function createMenuItem(payload) {
  const res = await fetch(`${API_BASE}/api/v1/menu/items`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

export async function patchMenuItem(id, updates) {
  const res = await fetch(`${API_BASE}/api/v1/menu/items/${id}`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify(updates),
  })
  return parseResponse(res)
}

export async function toggleMenuItemAvailability(id, available) {
  const res = await fetch(`${API_BASE}/api/v1/menu/items/${id}/availability`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ available }),
  })
  return parseResponse(res)
}

export async function uploadMenuItemImage(id, formData) {
  const token = localStorage.getItem('dzukku_token')
  const res = await fetch(`${API_BASE}/api/v1/menu/items/${id}/images`, {
    method: 'POST',
    headers: { Authorization: token ? `Bearer ${token}` : '' },
    body: formData,
  })
  return parseResponse(res)
}

// ── Tables / Sessions ────────────────────────────────────────────────────────

export async function fetchTables() {
  const res = await fetch(`${API_BASE}/api/v1/tables`, { headers: buildHeaders() })
  return parseResponse(res)
}

export async function createTable(payload) {
  const res = await fetch(`${API_BASE}/api/v1/tables`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

export async function updateTable(tableId, payload) {
  const res = await fetch(`${API_BASE}/api/v1/tables/${tableId}`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

export async function deleteTable(tableId) {
  const res = await fetch(`${API_BASE}/api/v1/tables/${tableId}`, {
    method: 'DELETE',
    headers: buildHeaders(),
  })
  if (res.status === 204) return { ok: true }
  return parseResponse(res)
}

export async function openTableSession(tableId, guests) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ table_id: tableId, guests }),
  })
  return parseResponse(res)
}

export async function fetchSessionOrders(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions/${sessionId}/orders`, { headers: buildHeaders() })
  return parseResponse(res)
}

export async function addTableSessionOrder(sessionId, items) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions/${sessionId}/orders`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ items }),
  })
  return parseResponse(res)
}

export async function fireTableSession(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions/${sessionId}/fire`, {
    method: 'POST',
    headers: buildHeaders(),
  })
  return parseResponse(res)
}

export async function generateTableInvoice(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions/${sessionId}/invoice`, {
    method: 'POST',
    headers: buildHeaders(),
  })
  return parseResponse(res)
}

export async function closeTableSession(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ status: 'CLOSED' }),
  })
  return parseResponse(res)
}

export async function fetchActiveSessions() {
  const res = await fetch(`${API_BASE}/api/v1/tables/sessions?status=OPEN`, { headers: buildHeaders() })
  return parseResponse(res)
}

// ── Kitchen ──────────────────────────────────────────────────────────────────

export async function fetchKitchenOrders() {
  const res = await fetch(`${API_BASE}/api/v1/kitchen/orders`, { headers: buildHeaders() })
  return parseResponse(res)
}

// ── Deliveries ───────────────────────────────────────────────────────────────

export async function fetchDeliveries(status) {
  const url = new URL(`${API_BASE}/api/v1/deliveries`)
  if (status) url.searchParams.set('status', status)
  const res = await fetch(url.toString(), { headers: buildHeaders() })
  return parseResponse(res)
}

export async function fetchDrivers() {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/drivers`, { headers: buildHeaders() })
  return parseResponse(res)
}

export async function assignDriver(orderId, driverId) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/assign`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ order_id: orderId, driver_id: driverId }),
  })
  return parseResponse(res)
}

export async function updateDeliveryStatus(deliveryId, status) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/${deliveryId}/status`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ status }),
  })
  return parseResponse(res)
}

export async function submitProofOfDelivery(deliveryId, proofUrl, proofType) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/${deliveryId}/proof`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ proof_url: proofUrl, proof_type: proofType }),
  })
  return parseResponse(res)
}

export async function updateDeliveryLocation(deliveryId, lat, lng) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/${deliveryId}/location`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ lat, lng }),
  })
  return parseResponse(res)
}

export async function trackDelivery(deliveryId) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/${deliveryId}/track`, { headers: buildHeaders() })
  return parseResponse(res)
}

export async function trackDeliveryByOrder(orderRef) {
  const res = await fetch(`${API_BASE}/api/v1/deliveries/orders/${orderRef}/track`, { headers: buildHeaders() })
  return parseResponse(res)
}

// ── Settlement ───────────────────────────────────────────────────────────────

export async function fetchBackendSettlement(orderId) {
  const res = await fetch(`${API_BASE}/api/v1/settlements/orders/${orderId}`, { headers: buildHeaders() })
  const data = await parseResponse(res)
  return data.fee_breakdown
}

// ── Reservations ─────────────────────────────────────────────────────────────

export async function fetchReservations(status) {
  const url = new URL(`${API_BASE}/api/v1/reservations`)
  if (status) url.searchParams.set('status', status)
  const res = await fetch(url.toString(), { headers: buildHeaders() })
  return parseResponse(res)
}

export async function updateReservationStatus(reservationId, status) {
  const res = await fetch(`${API_BASE}/api/v1/reservations/${reservationId}`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ status }),
  })
  return parseResponse(res)
}

// ── Staff ─────────────────────────────────────────────────────────────────────

export async function fetchStaff() {
  const res = await fetch(`${API_BASE}/api/v1/staff`, { headers: buildHeaders() })
  return parseResponse(res)
}

export async function createStaff(payload) {
  const res = await fetch(`${API_BASE}/api/v1/staff`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify(payload),
  })
  return parseResponse(res)
}

export async function toggleStaffActive(staffId, active) {
  const res = await fetch(`${API_BASE}/api/v1/staff/${staffId}/active`, {
    method: 'PATCH',
    headers: buildHeaders(),
    body: JSON.stringify({ active }),
  })
  return parseResponse(res)
}

// ── Invoices ──────────────────────────────────────────────────────────────────

export async function fetchInvoices() {
  const res = await fetch(`${API_BASE}/api/v1/invoices`, { headers: buildHeaders() })
  return parseResponse(res)
}
