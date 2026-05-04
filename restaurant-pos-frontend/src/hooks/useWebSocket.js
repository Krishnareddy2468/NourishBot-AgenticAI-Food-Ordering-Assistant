/**
 * Real-time WebSocket hook for Dzukku POS.
 * Connects to ws://host/api/v1/ws and dispatches events to callbacks.
 */

import { useEffect, useRef, useCallback, useState } from 'react'

const WS_BASE = (import.meta.env.VITE_WS_URL || '').replace(/^http/, 'ws') ||
  `ws://${window.location.hostname}:8000/api/v1/ws`

export function useWebSocket(restaurantId = 1) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const listenersRef = useRef({})

  const on = useCallback((eventType, callback) => {
    if (!listenersRef.current[eventType]) listenersRef.current[eventType] = []
    listenersRef.current[eventType].push(callback)
  }, [])

  const off = useCallback((eventType, callback) => {
    if (!listenersRef.current[eventType]) return
    listenersRef.current[eventType] = listenersRef.current[eventType].filter(cb => cb !== callback)
  }, [])

  useEffect(() => {
    let reconnectTimer = null

    function connect() {
      const url = `${WS_BASE}?restaurant_id=${restaurantId}`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          const eventType = data.event_type
          if (listenersRef.current[eventType]) {
            listenersRef.current[eventType].forEach(cb => cb(data))
          }
          // Wildcard listeners
          if (listenersRef.current['*']) {
            listenersRef.current['*'].forEach(cb => cb(data))
          }
        } catch { /* ignore malformed messages */ }
      }

      ws.onclose = () => {
        setConnected(false)
        reconnectTimer = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()
    return () => {
      clearTimeout(reconnectTimer)
      if (wsRef.current) wsRef.current.close()
    }
  }, [restaurantId])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, on, off, send }
}
