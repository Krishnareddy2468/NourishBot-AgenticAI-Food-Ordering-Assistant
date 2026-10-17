import './globals.css'
import Providers from './providers'

export const metadata = {
  title: 'NourishBot | Good meals, right on time',
  description: 'Plan meals, discover relevant restaurants, and schedule food around your day with NourishBot.',
}

export const viewport = {
  themeColor: '#FAF7F1',
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
