import './globals.css'
import Providers from './providers'

export const metadata = {
  title: 'Dzukku POS',
  description: 'Restaurant operations platform',
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
