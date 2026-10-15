import Link from 'next/link'
import NourishLogo from '../components/shared/NourishLogo'

export default function NotFound() {
  return (
    <main className="not-found-page">
      <NourishLogo />
      <span>404</span>
      <h1>This page isn’t on the menu.</h1>
      <p>The link may have moved, or the meal plan may no longer be available.</p>
      <Link href="/">Return home</Link>
    </main>
  )
}
