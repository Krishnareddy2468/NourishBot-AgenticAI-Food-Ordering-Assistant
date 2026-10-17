'use client'

import Image from 'next/image'
import Link from 'next/link'
import {
  ArrowRight,
  CalendarDays,
  CheckCircle2,
  Clock3,
  ShieldCheck,
} from 'lucide-react'
import EditorialSections from '../components/landing/EditorialSections'
import NourishLogo from '../components/shared/NourishLogo'

export default function HomePage() {
  return (
    <main className="nb-site">
      <header className="nb-nav-wrap">
        <nav className="nb-nav" aria-label="Primary navigation">
          <NourishLogo className="nourish-logo-header" />

          <div className="nb-nav-links">
            <a href="#how-it-works">How it works</a>
            <a href="#meal-planning">Home meals</a>
            <a href="#route-pickup">Route pickup</a>
            <a href="#chat-demo">Chat demo</a>
            <a href="#about">About</a>
          </div>

          <div className="nb-nav-actions">
            <Link className="nb-text-link" href="/login">Sign in</Link>
            <Link className="nb-pill-button nb-pill-dark" href="/login">
              Get started <ArrowRight size={15} />
            </Link>
          </div>
        </nav>
      </header>

      <section className="nb-hero" aria-labelledby="hero-title">
        <div className="nb-hero-copy">
          <div className="nb-kicker">Two ways to make food fit your day</div>
          <h1 id="hero-title">Eat well at home.<br /><em>Pick up on the way.</em></h1>
          <p>
            Plan a nutrition-aware day of home-delivered meals or find a practical pickup along
            your route, right inside WhatsApp and Telegram.
          </p>
          <div className="nb-hero-actions">
            <Link className="nb-button nb-button-primary" href="/login">
              Explore NourishBot <ArrowRight size={17} />
            </Link>
            <a className="nb-button nb-button-secondary" href="#how-it-works">
              See how it works
            </a>
          </div>
          <div className="nb-hero-note">
            <ShieldCheck size={17} />
            <span><strong>You stay in control.</strong> A plan is never an order. You review and confirm every checkout.</span>
          </div>
        </div>

        <div className="nb-hero-visual">
          <div className="nb-photo-frame">
            <Image
              src="/nourishbot-hero-v2.jpg"
              alt="A balanced bowl with millet, paneer, roasted vegetables, and greens"
              fill
              priority
              sizes="(max-width: 900px) 92vw, 48vw"
            />
            <span className="nb-photo-caption">A complete day, planned around you.</span>
          </div>
          <div className="nb-floating-plan" aria-label="Illustrative meal plan preview">
            <div className="nb-floating-plan-icon"><CalendarDays size={19} /></div>
            <div><small>UP NEXT</small><strong>Tomorrow’s lunch</strong><span>12:45 PM · Home delivery</span></div>
            <CheckCircle2 size={19} />
          </div>
          <div className="nb-time-stamp"><Clock3 size={14} /> Planned for your schedule</div>
        </div>
      </section>

      <EditorialSections />
    </main>
  )
}
