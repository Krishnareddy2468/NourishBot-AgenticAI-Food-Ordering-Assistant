import './globals.css'
import Providers from './providers'

export const metadata = {
  title: 'NourishBot — AI Scheduled Meal Assistant',
  description: 'Schedule Swiggy, Zomato, or NourishBot meals with route-aware restaurant discovery',
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
