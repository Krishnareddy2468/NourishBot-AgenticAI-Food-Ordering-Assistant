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
            <a href="#discover">Discover</a>
            <a href="#meal-planning">Meal planning</a>
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
          <div className="nb-kicker">Meals that fit your day</div>
          <h1 id="hero-title">Good meals.<br /><em>Right on time.</em></h1>
          <p>
            Meet NourishBot, your AI companion for planning meals, discovering great restaurants,
            and scheduling food around your day.
          </p>
          <div className="nb-hero-actions">
            <Link className="nb-button nb-button-primary" href="/login">
              Plan your meal <ArrowRight size={17} />
            </Link>
            <a className="nb-button nb-button-secondary" href="#how-it-works">
              See how it works
            </a>
          </div>
          <div className="nb-hero-note">
            <ShieldCheck size={17} />
            <span><strong>You stay in control.</strong> NourishBot plans and verifies before any supported handoff.</span>
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
            <span className="nb-photo-caption">Fresh choices, planned calmly.</span>
          </div>
          <div className="nb-floating-plan" aria-label="Illustrative meal plan preview">
            <div className="nb-floating-plan-icon"><CalendarDays size={19} /></div>
            <div><small>UP NEXT</small><strong>Tomorrow’s lunch</strong><span>12:45 PM · Pickup</span></div>
            <CheckCircle2 size={19} />
          </div>
          <div className="nb-time-stamp"><Clock3 size={14} /> Planned for your schedule</div>
        </div>
      </section>

      <EditorialSections />
    </main>
  )
}
