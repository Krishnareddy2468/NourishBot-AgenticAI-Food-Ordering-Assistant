'use client'
/**
 * Real-time WebSocket hook for Dzukku POS.
 * Connects to ws://host/api/v1/ws and dispatches events to callbacks.
 */

import { useEffect, useRef, useCallback, useState } from 'react'

const WS_BASE = (process.env.NEXT_PUBLIC_WS_URL || '').replace(/^http/, 'ws') ||
  (typeof window !== 'undefined' ? `ws://${window.location.hostname}:8000/api/v1/ws` : '')

export function useWebSocket(restaurantId = 1) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const listenersRef = useRef({})

  const on = useCallback((eventType, callback) => {
    if (!listenersRef.current[eventType]) listenersRef.current[eventType] = new Set()
    listenersRef.current[eventType].add(callback)
    return () => {
      listenersRef.current[eventType]?.delete(callback)
      if (listenersRef.current[eventType]?.size === 0) {
        delete listenersRef.current[eventType]
      }
    }
  }, [])

  const off = useCallback((eventType, callback) => {
    if (!listenersRef.current[eventType]) return
    listenersRef.current[eventType].delete(callback)
    if (listenersRef.current[eventType].size === 0) {
      delete listenersRef.current[eventType]
    }
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
      listenersRef.current = {}
    }
  }, [restaurantId])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, on, off, send }
}
