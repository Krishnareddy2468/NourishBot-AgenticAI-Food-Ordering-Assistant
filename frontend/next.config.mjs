import path from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = path.dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  outputFileTracingRoot: projectRoot,
  async headers() {
    const isDev = process.env.NODE_ENV !== 'production'
    const scriptPolicy = isDev ? "script-src 'self' 'unsafe-inline' 'unsafe-eval'" : "script-src 'self' 'unsafe-inline'"
    return [{
      source: '/:path*',
      headers: [
        { key: 'Content-Security-Policy', value: `default-src 'self'; ${scriptPolicy}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; font-src 'self' data:; connect-src 'self' http: https: ws: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'` },
        { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
        { key: 'X-Content-Type-Options', value: 'nosniff' },
        { key: 'X-Frame-Options', value: 'DENY' },
        { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=(self)' },
      ],
    }]
  },
}

export default nextConfig
