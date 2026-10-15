'use client'

import Image from 'next/image'
import Link from 'next/link'
import { useState } from 'react'
import {
  ArrowRight,
  CalendarDays,
  Check,
  CheckCircle2,
  Clock3,
  IndianRupee,
  MapPin,
  Navigation,
  Route,
  Search,
  ShieldCheck,
} from 'lucide-react'
import NourishLogo from '../shared/NourishLogo'

const steps = [
  ['01', 'Tell NourishBot what you want.', 'Share a craving, dietary preference, budget, and the time that suits you.'],
  ['02', 'Discover meals that fit your schedule.', 'Narrow the search around location, fulfilment, and supported providers.'],
  ['03', 'Review and confirm your plan.', 'Check verified details and decide what should happen next.'],
]

const days = [
  ['Mon', '19'],
  ['Tue', '20'],
  ['Wed', '21'],
  ['Thu', '22'],
  ['Fri', '23'],
  ['Sat', '24'],
  ['Sun', '25'],
]

const mealPeriods = ['Breakfast', 'Lunch', 'Snacks', 'Dinner']
const cuisines = ['All', 'South Indian', 'Bowls', 'Vegetarian']
const locations = ['Indiranagar', 'Koramangala', 'Near me']

export default function EditorialSections() {
  const [cuisine, setCuisine] = useState('All')
  const [fulfilment, setFulfilment] = useState('Pickup')
  const [mealTime, setMealTime] = useState('1:00 PM')
  const [discoveryView, setDiscoveryView] = useState('list')
  const [activeDay, setActiveDay] = useState('21')
  const [mealPeriod, setMealPeriod] = useState('Lunch')
  const [chatLocation, setChatLocation] = useState('')

  return (
    <>
      <section className="ed-process" id="how-it-works" aria-labelledby="ed-process-title">
        <div className="ed-process-copy">
          <span className="ed-label">HOW IT WORKS</span>
          <h2 id="ed-process-title">Your next great meal, already planned.</h2>
          <p className="ed-lead">A clear path from a passing craving to a meal that genuinely fits the day ahead.</p>

          <ol className="ed-steps">
            {steps.map(([number, title, copy]) => (
              <li key={number}>
                <span>{number}</span>
                <div><h3>{title}</h3><p>{copy}</p></div>
              </li>
            ))}
          </ol>
        </div>

        <div className="ed-process-visual">
          <div className="ed-process-photo">
            <Image
              src="/nourishbot-process-v1.jpg"
              alt="South Indian lunch with lemon rice, vegetables, and dal"
              fill
              sizes="(max-width: 820px) 92vw, 44vw"
            />
          </div>
          <div className="ed-schedule-slip" aria-label="Illustrative scheduling preview">
            <div className="ed-slip-top"><span>MEAL PLAN</span><small>Preview</small></div>
            <strong>Tomorrow’s lunch</strong>
            <div className="ed-slip-row"><Clock3 size={14} /><span>12:45 PM</span><i /><span>Pickup</span></div>
            <div className="ed-slip-status"><CheckCircle2 size={14} /> Ready for review</div>
          </div>
          <span className="ed-photo-note">A calm lunch, before the calendar gets loud.</span>
        </div>
      </section>

      <section className="ed-discovery" id="discover" aria-labelledby="ed-discovery-title">
        <header className="ed-discovery-heading">
          <div><span className="ed-label">RESTAURANT DISCOVERY</span><h2 id="ed-discovery-title">Find food worth looking forward to.</h2></div>
          <p>Discover restaurants that fit your taste, budget, location, and schedule.</p>
        </header>

        <div className="ed-discovery-stage">
          <div className="ed-discovery-photo">
            <Image
              src="/nourishbot-discovery-v1.jpg"
              alt="Crisp masala dosa served with sambar and coconut chutney in a sunlit restaurant"
              fill
              sizes="(max-width: 900px) 100vw, 58vw"
            />
            <span>South Indian · Demo photography</span>
          </div>

          <div className="ed-discovery-panel">
            <div className="ed-preview-note">INTERACTIVE PREVIEW · LIVE DATA NOT CONNECTED</div>
            <label className="ed-search">
              <Search size={16} />
              <span className="sr-only">Discovery location</span>
              <input readOnly value="Indiranagar, Bengaluru" aria-label="Preview discovery location" />
            </label>

            <div className="ed-control-pair">
              <div>
                <small>FULFILMENT</small>
                <div className="ed-choice-row">
                  {['Pickup', 'Delivery'].map((item) => <button key={item} type="button" aria-pressed={fulfilment === item} className={fulfilment === item ? 'active' : ''} onClick={() => setFulfilment(item)}>{item}</button>)}
                </div>
              </div>
              <label className="ed-time-select"><small>MEAL TIME</small><select value={mealTime} onChange={(event) => setMealTime(event.target.value)} aria-label="Preview meal time"><option>12:30 PM</option><option>1:00 PM</option><option>1:30 PM</option></select></label>
            </div>

            <div className="ed-cuisine-row" aria-label="Cuisine filters">
              {cuisines.map((item) => <button key={item} type="button" aria-pressed={cuisine === item} className={cuisine === item ? 'active' : ''} onClick={() => setCuisine(item)}>{item}</button>)}
            </div>

            <div className="ed-view-switch" aria-label="Discovery view">
              <button type="button" className={discoveryView === 'list' ? 'active' : ''} aria-pressed={discoveryView === 'list'} onClick={() => setDiscoveryView('list')}>Recommendations</button>
              <button type="button" className={discoveryView === 'map' ? 'active' : ''} aria-pressed={discoveryView === 'map'} onClick={() => setDiscoveryView('map')}>Map</button>
            </div>

            {discoveryView === 'list' ? (
              <div className="ed-recommendations">
                <article>
                  <span className="ed-result-index">A</span>
                  <div><small>DEMO RESULT</small><h3>Nearby {cuisine === 'All' ? 'meal' : cuisine.toLowerCase()} option</h3><p>{fulfilment} at {mealTime} · availability verified when connected</p></div>
                  <ArrowRight size={16} />
                </article>
                <article>
                  <span className="ed-result-index">B</span>
                  <div><small>DEMO RESULT</small><h3>Schedule-friendly alternative</h3><p>Price and provider shown only after verification</p></div>
                  <ArrowRight size={16} />
                </article>
              </div>
            ) : (
              <div className="ed-map-unavailable">
                <MapPin size={20} />
                <div><strong>Map view needs a configured Maps connection.</strong><p>Live markers, travel distance, and restaurant details will appear here when available.</p></div>
              </div>
            )}

            <div className="ed-verified-line"><ShieldCheck size={14} /> Prices, ratings, and availability are never invented.</div>
          </div>
        </div>
      </section>

      <section className="ed-planning" id="meal-planning" aria-labelledby="ed-planning-title">
        <header className="ed-planning-heading">
          <span className="ed-label">MEAL PLANNING</span>
          <h2 id="ed-planning-title">A little planning.<br />A lot more living.</h2>
          <p>Make meals fit your day, not the other way around.</p>
        </header>

        <div className="ed-planning-layout">
          <aside className="ed-planning-aside">
            <div className="ed-planning-photo">
              <Image src="/nourishbot-process-v1.jpg" alt="A planned South Indian lunch" fill sizes="(max-width: 860px) 100vw, 32vw" />
            </div>
            <div className="ed-planning-chat">
              <small>PLANNING NOTE</small>
              <p>“Keep Wednesday lunch light and vegetarian.”</p>
              <span>NourishBot will use this when you start a real plan.</span>
            </div>
          </aside>

          <div className="ed-planner-ui">
            <div className="ed-planner-top">
              <div><small>YOUR WEEK</small><h3>October 19–25</h3></div>
              <span>Product preview</span>
            </div>

            <div className="ed-week" aria-label="Select a day">
              {days.map(([day, date]) => (
                <button key={date} type="button" aria-pressed={activeDay === date} className={activeDay === date ? 'active' : ''} onClick={() => setActiveDay(date)}><small>{day}</small><strong>{date}</strong><i /></button>
              ))}
            </div>

            <div className="ed-periods" aria-label="Select a meal period">
              {mealPeriods.map((period) => <button key={period} type="button" aria-pressed={mealPeriod === period} className={mealPeriod === period ? 'active' : ''} onClick={() => setMealPeriod(period)}>{period}</button>)}
            </div>

            <div className="ed-day-detail">
              <div className="ed-day-time"><span>{mealPeriod}</span><strong>{mealPeriod === 'Breakfast' ? '8:30' : mealPeriod === 'Lunch' ? '12:45' : mealPeriod === 'Snacks' ? '4:30' : '7:45'} <small>{mealPeriod === 'Breakfast' ? 'AM' : 'PM'}</small></strong></div>
              <div className="ed-day-copy"><small>{days.find(([, date]) => date === activeDay)?.[0]}, OCTOBER {activeDay}</small><h4>No meal confirmed yet</h4><p>Choose a meal and verify a supported time before anything is scheduled.</p></div>
              <Link href="/login">Plan this meal <ArrowRight size={15} /></Link>
            </div>

            <div className="ed-upcoming-row">
              <span><CalendarDays size={16} /></span>
              <div><strong>Upcoming meals will appear here</strong><p>Your account currently has no verified schedule in this preview.</p></div>
              <Check size={15} />
            </div>
          </div>
        </div>
      </section>

      <section className="ed-route" aria-labelledby="ed-route-title">
        <div className="ed-route-copy">
          <span className="ed-label">SMART ROUTE DISCOVERY</span>
          <h2 id="ed-route-title">Good food, along the way.</h2>
          <p>Find a practical pickup stop between where you are and where you need to be.</p>
          <div className="ed-route-disclosure"><ShieldCheck size={15} /> Preview only—detour and eligibility require connected route data.</div>
        </div>

        <div className="ed-route-product">
          <div className="ed-route-fields">
            <label><small>FROM</small><span><MapPin size={14} /> Home</span></label>
            <i />
            <label><small>TO</small><span><Navigation size={14} /> Office</span></label>
            <label><small>COLLECT</small><span><Clock3 size={14} /> 12:45 PM</span></label>
          </div>
          <div className="ed-route-track" aria-label="Illustrative route from home to office with a possible food stop">
            <span className="ed-route-origin">Home</span><i /><b><Route size={16} /></b><i /><span className="ed-route-stop">Food stop<small>Verify pickup</small></span><i /><span className="ed-route-destination">Office</span>
          </div>
          <div className="ed-route-result">
            <span className="ed-result-index">A</span>
            <div><small>DEMO CANDIDATE</small><h3>Pickup option on your route</h3><p>Estimated detour appears when Maps is connected.</p></div>
            <div className="ed-eligibility"><span>Regular pickup</span><small>Drive-through unverified</small></div>
          </div>
        </div>
      </section>

      <section className="ed-conversation" aria-labelledby="ed-conversation-title">
        <div className="ed-conversation-copy">
          <span className="ed-label">CONVERSATIONAL PLANNING</span>
          <h2 id="ed-conversation-title">Just tell NourishBot.</h2>
          <p>Begin with the way you naturally think about food. NourishBot asks for what is missing, then turns the answer into a plan you can review.</p>
          <Link href="/login" className="ed-text-link">Start a conversation <ArrowRight size={16} /></Link>
        </div>

        <div className="ed-chat-ui">
          <div className="ed-chat-head"><div className="ed-chat-monogram">N</div><div><strong>NourishBot</strong><small>Planning assistant · Preview</small></div></div>
          <div className="ed-chat-body">
            <div className="ed-message ed-message-user">Find me a vegetarian lunch tomorrow under ₹200.</div>
            <div className="ed-message ed-message-bot">I can help you find suitable options. What location should I search?</div>
            <div className="ed-location-choices" aria-label="Choose a preview location">
              {locations.map((location) => <button key={location} type="button" aria-pressed={chatLocation === location} className={chatLocation === location ? 'active' : ''} onClick={() => setChatLocation(location)}>{location}</button>)}
            </div>
            {chatLocation && <div className="ed-chat-next"><MapPin size={15} /><span><strong>{chatLocation} selected</strong><small>Continue to verify live restaurant options.</small></span><Link href="/login" aria-label="Continue this plan"><ArrowRight size={15} /></Link></div>}
          </div>
          <div className="ed-chat-input"><span>Describe your next meal…</span><Link href="/login" aria-label="Open NourishBot"><ArrowRight size={16} /></Link></div>
        </div>
      </section>

      <section className="ed-final" id="about" aria-labelledby="ed-final-title">
        <Image src="/nourishbot-table-v1.jpg" alt="A warm Indian dinner table with curry, rice, roti, and salad" fill sizes="100vw" />
        <div className="ed-final-shade" />
        <div className="ed-final-copy">
          <span className="ed-label">PLAN AHEAD, EAT WELL</span>
          <h2 id="ed-final-title">Tomorrow’s meal starts today.</h2>
          <p>Spend less time deciding. More time enjoying.</p>
          <Link href="/login" className="ed-final-button">Start planning <ArrowRight size={17} /></Link>
        </div>
      </section>

      <footer className="ed-footer">
        <NourishLogo className="nourish-logo-footer" tagline="Meals that fit your day." />
        <nav aria-label="Footer navigation"><a href="#how-it-works">How it works</a><a href="#discover">Discover</a><a href="#meal-planning">Meal planning</a><Link href="/login">Sign in</Link></nav>
        <p>Discovery information is verified through connected services where available. A plan or provider handoff is not a placed order until you explicitly confirm it.</p>
        <span className="ed-footer-copy">© 2026 NourishBot</span>
      </footer>
    </>
  )
}
