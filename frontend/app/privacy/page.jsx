import Link from 'next/link'
import NourishLogo from '../../components/shared/NourishLogo'

export const metadata = {
  title: 'Privacy | NourishBot',
  description: 'How NourishBot uses, retains, exports, and deletes customer information.',
}

export default function PrivacyPage() {
  return <main className="privacy-page">
    <header><NourishLogo /><Link href="/">Back to home</Link></header>
    <article>
      <span>PRIVACY AT NOURISHBOT</span>
      <h1>Your meal data stays under your control.</h1>
      <p className="privacy-intro">Effective 17 October 2026. This notice describes the product’s data controls. The production operator must publish its legal identity and privacy contact before enabling public transactional release.</p>

      <section><h2>What we process</h2><p>We process account and verified channel identifiers, meal plans, orders, delivery or pickup details, consent decisions, and operational records needed to provide the service. Location, personalization, reminder, and analytics data are used only when the corresponding consent is active.</p></section>
      <section><h2>Why we process it</h2><p>Information is used to plan meals, verify restaurants and fulfillment, schedule confirmed orders, communicate status, prevent duplicate transactions, reconcile provider outcomes, enforce budgets, and protect the service. Dietary preferences are not treated as medical advice.</p></section>
      <section><h2>Providers and payments</h2><p>Google Maps is used only for discovery. Swiggy and Zomato remain in discovery or user-controlled handoff mode unless their exact transactional capabilities are verified. Provider receipts are redacted. Payment providers receive the minimum information required for payment processing.</p></section>
      <section><h2>Retention</h2><p>Chat turns, shared locations, route history, and provider excerpts have bounded retention periods configured by the operator. Consent and transaction audit records are retained when required for security, accounting, dispute resolution, or legal obligations.</p></section>
      <section><h2>Your controls</h2><p>In the customer account you can change purpose-specific consent, update preferences, and inspect plans, orders, and spending. Linked chat channels support <code>privacy status</code>, <code>export my memory</code>, and the confirmed deletion workflow. Disabling location consent removes retained location and route data.</p></section>
      <section><h2>Security and deletion</h2><p>Customer and tenant scope comes from verified identities rather than request parameters. Sensitive cookies are HttpOnly, provider payloads are redacted, and transaction histories are append-only. A permanent deletion request requires explicit confirmation and is recorded for audit.</p></section>
      <aside>For privacy assistance, use the authenticated privacy controls in your NourishBot account or the support channel published by your deployment operator.</aside>
    </article>
  </main>
}
