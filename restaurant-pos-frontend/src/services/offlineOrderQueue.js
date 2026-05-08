const CACHE_KEYS = {
  tables: 'dzukku_waiter_tables_cache',
  sessions: 'dzukku_waiter_sessions_cache',
  menu: 'dzukku_waiter_menu_cache',
  queue: 'dzukku_waiter_offline_queue_v1',
}

function readJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function writeJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value))
}

export function getWaiterCache() {
  return {
    tables: readJson(CACHE_KEYS.tables, []),
    sessions: readJson(CACHE_KEYS.sessions, []),
    menu: readJson(CACHE_KEYS.menu, []),
  }
}

export function setWaiterCache(data) {
  if (data.tables) writeJson(CACHE_KEYS.tables, data.tables)
  if (data.sessions) writeJson(CACHE_KEYS.sessions, data.sessions)
  if (data.menu) writeJson(CACHE_KEYS.menu, data.menu)
}

export function getOfflineOrderQueue() {
  return readJson(CACHE_KEYS.queue, [])
}

export function setOfflineOrderQueue(queue) {
  writeJson(CACHE_KEYS.queue, queue)
}

export function enqueueOfflineOrder(entry) {
  const queue = getOfflineOrderQueue()
  const next = [...queue, entry]
  setOfflineOrderQueue(next)
  return next
}

export function removeOfflineOrder(queueId) {
  const next = getOfflineOrderQueue().filter(entry => entry.queueId !== queueId)
  setOfflineOrderQueue(next)
  return next
}

export function updateOfflineOrder(queueId, updates) {
  const next = getOfflineOrderQueue().map(entry => (
    entry.queueId === queueId ? { ...entry, ...updates } : entry
  ))
  setOfflineOrderQueue(next)
  return next
}

export function buildOfflineOrderEntry({ sessionId, table, items, guests }) {
  const createdAt = new Date().toISOString()
  return {
    queueId: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `offline-${Date.now()}`,
    sessionId,
    tableId: table?.id || null,
    tableName: table?.table_number || table?.name || `Table ${table?.id || ''}`.trim(),
    guests: guests || null,
    createdAt,
    status: 'QUEUED',
    items: items.map(item => ({
      id: item.id,
      name: item.name,
      qty: item.qty,
      price_cents: item.price_cents,
      category_name: item.category_name || item.category || '',
    })),
  }
}
