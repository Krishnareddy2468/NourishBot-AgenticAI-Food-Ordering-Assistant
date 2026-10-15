import Link from 'next/link'
import {
  ArrowRight,
  Check,
  ChefHat,
  CarFront,
  Clock3,
  LayoutDashboard,
  MapPin,
  MessageCircle,
  Navigation,
  ShieldCheck,
  ShoppingBag,
  Store,
  UtensilsCrossed,
  Truck,
} from 'lucide-react'

const flow = [
  {
    number: '01',
    icon: MessageCircle,
    title: 'Chat naturally',
    copy: 'Start on Telegram or WhatsApp and describe the meal or dietary preference you want.',
  },
  {
    number: '02',
    icon: Store,
    title: 'Choose where to order',
    copy: 'Schedule from NourishBot, Swiggy, or Zomato. No unrelated providers are shown.',
  },
  {
    number: '03',
    icon: Navigation,
    title: 'Search by place or route',
    copy: 'Find well-rated restaurants near you or drive-through options along your driving route.',
  },
  {
    number: '04',
    icon: ShoppingBag,
    title: 'Review and schedule',
    copy: 'Choose home delivery or drive-through pickup, set a future time, review, and confirm.',
  },
]

const mealPeriods = [
  { number: '01', name: 'Breakfast', note: 'Start fresh', time: '7–10 AM' },
  { number: '02', name: 'Lunch', note: 'Beat the rush', time: '12–3 PM' },
  { number: '03', name: 'Snacks', note: 'Quick recharge', time: '4–6 PM' },
  { number: '04', name: 'Dinner', note: 'End the day well', time: '7–10 PM' },
]

const foodPreferences = [
  'Vegetarian',
  'Vegan',
  'High-protein',
  'Lighter meals',
  'Balanced plates',
  'Low-sugar options',
]

export default function HomePage() {
  const workspaceHref = '/workspace'
  const workspaceLabel = 'Open workspace'

  return (
    <main className="landing-page">
      <nav className="landing-nav" aria-label="Primary navigation">
        <Link className="landing-brand" href="/" aria-label="NourishBot home">
          <span className="landing-brand-mark"><UtensilsCrossed size={20} /></span>
          <span>NourishBot</span>
        </Link>

        <div className="landing-nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="#meal-options">Meal options</a>
          <a href="#safety">Safety</a>
        </div>

        <Link className="landing-nav-cta" href={workspaceHref}>
          {workspaceLabel} <ArrowRight size={15} />
        </Link>
      </nav>

      <section className="landing-hero">
        <div className="landing-hero-copy">
          <div className="landing-eyebrow">
            <Clock3 size={14} /> Meal scheduling · Telegram + WhatsApp
          </div>
          <h1>Plan every meal.<br /><span>Before the rush.</span></h1>
          <p>
            Chat through Telegram or WhatsApp, discover restaurants with Maps, and schedule
            breakfast, lunch, snacks, or dinner through NourishBot, Swiggy, or Zomato.
          </p>
          <div className="landing-hero-actions">
            <a className="landing-button landing-button-primary" href="#how-it-works">
              See how it works <ArrowRight size={18} />
            </a>
            <Link className="landing-button landing-button-secondary" href={workspaceHref}>
              <LayoutDashboard size={18} /> {workspaceLabel}
            </Link>
          </div>
          <div className="landing-proof" aria-label="Product boundaries">
            <span><Check size={14} /> No instant ordering</span>
            <span><Check size={14} /> Swiggy + Zomato ordering</span>
            <span><Check size={14} /> Human-confirmed actions</span>
            <span><Check size={14} /> Pickup, drive-through, or delivery</span>
          </div>
        </div>

        <div className="landing-demo" aria-label="NourishBot order preview">
          <div className="landing-demo-topbar">
            <div className="landing-demo-title">
              <span className="landing-demo-logo"><UtensilsCrossed size={16} /></span>
              <div><strong>NourishBot</strong><small>Campus canteen assistant</small></div>
            </div>
            <span className="landing-online"><i /> Online</span>
          </div>

          <div className="landing-chat">
            <div className="landing-message landing-message-user">
              Schedule a healthy vegetarian dinner tomorrow at 7:30 PM for home delivery.
            </div>
            <div className="landing-message landing-message-bot">
              <span className="landing-message-icon"><UtensilsCrossed size={13} /></span>
              <div>
                <p>Here is a balanced combo from today&apos;s available menu:</p>
                <div className="landing-order-card">
                  <div><span>Paneer millet bowl</span><strong>₹169</strong></div>
                  <div><span>Fresh lime water</span><strong>₹35</strong></div>
                  <div className="landing-order-total"><span>Total</span><strong>₹204</strong></div>
                </div>
                <button type="button" tabIndex={-1}>Confirm schedule <ArrowRight size={14} /></button>
              </div>
            </div>
          </div>

          <div className="landing-status-card">
            <div className="landing-status-heading">
              <span><Clock3 size={16} /> Order NB-1042</span>
              <strong>Scheduled</strong>
            </div>
            <div className="landing-progress"><span /></div>
            <div className="landing-status-labels"><span>Tomorrow · 7:30 PM</span><span><MapPin size={10} /> Home delivery</span></div>
          </div>
        </div>
      </section>

      <section className="landing-flow" id="how-it-works">
        <div className="landing-section-heading">
          <span>A calmer ordering flow</span>
          <h2>From conversation to a confirmed schedule</h2>
          <p>Maps helps with discovery; the selected ordering provider verifies the final restaurant, menu, and slot.</p>
        </div>

        <div className="landing-flow-grid">
          {flow.map(({ number, icon: Icon, title, copy }) => (
            <article className="landing-flow-card" key={number}>
              <div className="landing-flow-meta"><span>{number}</span><Icon size={22} /></div>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-provider-band" aria-label="Supported channels and providers">
        <div><small>CHAT CHANNELS</small><strong>Telegram</strong><strong>WhatsApp</strong></div>
        <span />
        <div><small>ORDERING</small><strong>NourishBot</strong><strong>Swiggy</strong><strong>Zomato</strong></div>
        <span />
        <div><small>MAPS + FULFILLMENT</small><strong>Nearby search</strong><strong>On-route drive-through</strong><strong>Home delivery</strong></div>
      </section>

      <section className="landing-choice" id="meal-options" aria-labelledby="meal-choice-title">
        <div className="landing-choice-heading">
          <span>Made for your whole day</span>
          <h2 id="meal-choice-title">Good food, whenever you need it.</h2>
          <p>Browse by mealtime or ask NourishBot to narrow the menu around your preferences.</p>
        </div>

        <div className="landing-meal-grid">
          {mealPeriods.map((meal) => (
            <article className="landing-meal-card" key={meal.name}>
              <span className="landing-meal-index">{meal.number}</span>
              <div className="landing-meal-copy"><strong>{meal.name}</strong><small>{meal.note}</small></div>
              <time>{meal.time}</time>
            </article>
          ))}
        </div>

        <div className="landing-preference-panel">
          <div>
            <span>Preference-led discovery</span>
            <h3>Find food that fits your routine</h3>
          </div>
          <div className="landing-preference-chips">
            {foodPreferences.map((preference) => <span key={preference}><Check size={13} /> {preference}</span>)}
          </div>
          <p>Suggestions use available menu metadata and personal preferences. They are not medical or allergy-safety advice.</p>
        </div>
      </section>

      <section className="landing-product" id="product">
        <div className="landing-product-copy">
          <div className="landing-eyebrow"><Navigation size={14} /> Route-aware food discovery</div>
          <h2>Find a practical stop, not just a popular restaurant.</h2>
          <p>
            NourishBot can compare restaurants near your location or along the route you are
            already driving, then verify the chosen option with Swiggy or Zomato.
          </p>
          <ul className="landing-feature-list">
            <li><MessageCircle size={18} /><div><strong>Two chat channels</strong><span>Plan through Telegram or the verified WhatsApp webhook.</span></div></li>
            <li><ChefHat size={18} /><div><strong>Three clear ordering choices</strong><span>Use NourishBot, Swiggy, or Zomato—no placeholder providers.</span></div></li>
            <li><ShoppingBag size={18} /><div><strong>Mandatory scheduling</strong><span>Reject ASAP requests and require a future meal slot.</span></div></li>
            <li><CarFront size={18} /><div><strong>Drive-through along your route</strong><span>Maps ranks route-side candidates; the ordering app verifies pickup support.</span></div></li>
            <li><Truck size={18} /><div><strong>Scheduled home delivery</strong><span>NourishBot coordinates the address and time; the selected network fulfils the order.</span></div></li>
          </ul>
        </div>

        <div className="landing-ops-card">
          <div className="landing-ops-header">
            <div><small>ROUTE SEARCH</small><h3>Drive-through on your way</h3></div>
            <span><i /> Maps ready</span>
          </div>
          <div className="landing-ops-stats">
            <div><small>Route</small><strong>18</strong><span>km total</span></div>
            <div><small>Options</small><strong>03</strong><span>Low detour</span></div>
            <div><small>Meal time</small><strong>1:00</strong><span>Tomorrow</span></div>
          </div>
          <div className="landing-ops-orders">
            <div><span className="landing-order-id">4.6 ★</span><span>Highway Kitchen · +4 min</span><strong className="landing-tag cooking">Verify</strong></div>
            <div><span className="landing-order-id">4.4 ★</span><span>Green Bowl · +7 min</span><strong className="landing-tag ready">Swiggy</strong></div>
            <div><span className="landing-order-id">4.3 ★</span><span>South Stop · +9 min</span><strong className="landing-tag queued">Zomato</strong></div>
          </div>
        </div>
      </section>

      <section className="landing-safety" id="safety">
        <div className="landing-safety-icon"><ShieldCheck size={30} /></div>
        <div>
          <span>Clear operating boundaries</span>
          <h2>The assistant proposes. Policy verifies the order.</h2>
          <p>
            NourishBot never lets generated text directly change persistent order state.
            Deterministic checks validate the future time, provider capability, availability,
            quantities, totals, and confirmation before any action is committed.
          </p>
        </div>
        <div className="landing-safety-pipeline" aria-label="Safety pipeline">
          <span>AI proposes</span><ArrowRight size={16} /><span>Schedule checks</span><ArrowRight size={16} /><span>Provider commits</span>
        </div>
      </section>

      <section className="landing-final-cta">
        <span className="landing-final-mark"><UtensilsCrossed size={26} /></span>
        <h2>Your meals, planned before you need them.</h2>
        <p>Breakfast, lunch, snacks, or dinner—choose the provider, fulfillment method, and future time.</p>
        <Link className="landing-button landing-button-primary" href={workspaceHref}>
          {workspaceLabel} <ArrowRight size={18} />
        </Link>
      </section>

      <footer className="landing-footer">
        <Link className="landing-brand" href="/"><span className="landing-brand-mark"><UtensilsCrossed size={18} /></span><span>NourishBot</span></Link>
        <p>Scheduled-meal ordering · Telegram and WhatsApp</p>
        <span>No instant ordering</span>
      </footer>
    </main>
  )
}
