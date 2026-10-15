import './globals.css'
import Providers from './providers'

export const metadata = {
  title: 'NourishBot — AI Canteen Pre-Order',
  description: 'AI-assisted canteen pre-order, kitchen, and smart-pickup workspace',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
