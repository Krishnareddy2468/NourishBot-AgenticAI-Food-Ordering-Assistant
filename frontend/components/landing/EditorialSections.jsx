'use client'

import Image from 'next/image'
import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import {
  ArrowRight,
  CalendarDays,
  Check,
  CheckCircle2,
  Clock3,
  HeartPulse,
  Home,
  MapPin,
  MessageCircle,
  Navigation,
  Route,
  Search,
  Send,
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
const chatDemos = {
  meals: {
    label: 'Home meal plan',
    eyebrow: 'FEATURE 01 · NUTRITION-AWARE HOME MEALS',
    title: 'Build a full day of meals around your life.',
    description: 'Set a diet, budget and nutrition target. NourishBot prepares four editable meal slots that you can review before ordering anything.',
  },
  route: {
    label: 'On-route pickup',
    eyebrow: 'FEATURE 02 · ON-ROUTE PICKUP',
    title: 'Find a practical food stop on your way.',
    description: 'Share your route, compare useful stops and choose a future pickup time. NourishBot checks provider support before any order can continue.',
  },
}

const channelConversations = {
  whatsapp: {
    meals: {
      first: 'Vegetarian meals tomorrow, under ₹800. Aim for 90g protein.',
      reply: 'Here is a four-meal draft. Tap a meal to change it.',
      final: 'This is a draft to review. No meal has been ordered. Each order needs its own details and explicit confirmation.',
    },
    route: {
      first: 'I’m driving from Indiranagar to Whitefield. Find a vegetarian pickup on my route.',
      reply: 'I found two practical stops. Detours and arrival times are estimates, not food-ready guarantees.',
      final: 'This stop is ready for review. Provider support, the future time, item and price must be checked before any order confirmation.',
    },
  },
  telegram: {
    meals: {
      first: 'Plan tomorrow: vegetarian, ₹800 budget, 90g protein target.',
      reply: 'Here is your four-meal draft. Tap a meal to edit it.',
      final: 'This is a draft to review. No order was created. Continue through the individual meal order flow when you are ready.',
    },
    route: {
      first: 'Find veg pickup stops from Indiranagar to Whitefield for 1:15 PM.',
      reply: 'Two route candidates found. Travel estimates do not confirm restaurant preparation or readiness.',
      final: 'This route stop is for review. Provider pickup support and the future time still need verification.',
    },
  },
}

const mealSlots = [
  { slot: 'Breakfast', time: '8:00 AM', options: [{ name: 'Vegetable poha', kcal: 410, protein: 18, price: 150 }, { name: 'Paneer dosa', kcal: 520, protein: 24, price: 190 }] },
  { slot: 'Lunch', time: '12:45 PM', options: [{ name: 'Millet paneer bowl', kcal: 560, protein: 31, price: 240 }, { name: 'Rajma rice bowl', kcal: 590, protein: 23, price: 210 }] },
  { slot: 'Snack', time: '4:30 PM', options: [{ name: 'Fruit yoghurt cup', kcal: 220, protein: 12, price: 120 }, { name: 'Roasted chana chaat', kcal: 260, protein: 14, price: 110 }] },
  { slot: 'Dinner', time: '7:45 PM', options: [{ name: 'Tofu khichdi', kcal: 610, protein: 32, price: 250 }, { name: 'Palak paneer plate', kcal: 680, protein: 29, price: 280 }] },
]

const routeCandidates = [
  { name: 'Green Bowl Kitchen', food: 'Vegetarian grain bowls', area: 'Domlur', detour: '+6 min', arrival: '1:08 PM', pickup: '1:15 PM', status: 'Provider check required' },
  { name: 'Millet House', food: 'South Indian meals', area: 'Marathahalli', detour: '+9 min', arrival: '1:12 PM', pickup: '1:20 PM', status: 'Future pickup unverified' },
]

export default function EditorialSections() {
  const [cuisine, setCuisine] = useState('All')
  const [fulfilment, setFulfilment] = useState('Pickup')
  const [mealTime, setMealTime] = useState('1:00 PM')
  const [discoveryView, setDiscoveryView] = useState('list')
  const [activeDay, setActiveDay] = useState('21')
  const [mealPeriod, setMealPeriod] = useState('Lunch')
  const [chatFeature, setChatFeature] = useState('meals')
  const [chatChannel, setChatChannel] = useState('whatsapp')
  const [mealSelections, setMealSelections] = useState([0, 0, 0, 0])
  const [routeCandidate, setRouteCandidate] = useState(0)
  const [previewReviewed, setPreviewReviewed] = useState(false)
  const messagesRef = useRef(null)

  const activeDemo = chatDemos[chatFeature]
  const conversation = channelConversations[chatChannel][chatFeature]
  const activeRoute = routeCandidates[routeCandidate]
  const mealTotals = mealSlots.reduce((totals, meal, index) => {
    const selected = meal.options[mealSelections[index]]
    return {
      price: totals.price + selected.price,
      kcal: totals.kcal + selected.kcal,
      protein: totals.protein + selected.protein,
    }
  }, { price: 0, kcal: 0, protein: 0 })

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = previewReviewed ? messagesRef.current.scrollHeight : 0
    }
  }, [chatFeature, chatChannel, previewReviewed])

  const selectFeature = (feature) => {
    setChatFeature(feature)
    setPreviewReviewed(false)
  }

  const selectChannel = (channel) => {
    setChatChannel(channel)
    setPreviewReviewed(false)
  }

  const cycleMeal = (index) => {
    setMealSelections((current) => current.map((selection, mealIndex) => (
      mealIndex === index ? (selection + 1) % mealSlots[index].options.length : selection
    )))
    setPreviewReviewed(false)
  }

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
              <div><small>YOUR WEEK</small><h3>October 19 to 25</h3></div>
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

      <section className="ed-route" id="route-pickup" aria-labelledby="ed-route-title">
        <div className="ed-route-copy">
          <span className="ed-label">SMART ROUTE DISCOVERY</span>
          <h2 id="ed-route-title">Good food, along the way.</h2>
          <p>Find a practical pickup stop between where you are and where you need to be.</p>
          <div className="ed-route-disclosure"><ShieldCheck size={15} /> Preview only. Detour and eligibility require connected route data.</div>
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

      <section className="demo-chat-section" id="chat-demo" aria-labelledby="demo-chat-title">
        <header className="demo-chat-heading">
          <div>
            <span className="ed-label">WHATSAPP + TELEGRAM DEMO</span>
            <h2 id="demo-chat-title">Two everyday needs.<br />One natural conversation.</h2>
          </div>
          <p>Switch the feature and channel to preview how NourishBot guides the conversation. All content below is an illustrative product demo, not a live order.</p>
        </header>

        <div className="demo-chat-stage">
          <aside className="demo-feature-picker" aria-label="Choose a feature demo">
            {Object.entries(chatDemos).map(([key, demo], index) => (
              <button
                key={key}
                type="button"
                className={chatFeature === key ? 'active' : ''}
                aria-pressed={chatFeature === key}
                onClick={() => selectFeature(key)}
              >
                <span className="demo-feature-icon" aria-hidden="true">{index === 0 ? <HeartPulse size={19} /> : <Route size={19} />}</span>
                <span className="demo-feature-text"><strong>{demo.label}</strong><small>{index === 0 ? 'Four meals shaped by your day and budget.' : 'Compare food stops along your drive.'}</small></span>
                <span className="demo-feature-state">{chatFeature === key ? 'Selected' : 'Explore'}</span>
              </button>
            ))}

            <div className="demo-feature-copy" aria-live="polite">
              <span className="ed-label">{activeDemo.eyebrow}</span>
              <h3>{activeDemo.title}</h3>
              <p>{activeDemo.description}</p>
              <div><ShieldCheck size={15} /> Preview data is illustrative. Live details require checks.</div>
            </div>
          </aside>

          <div className="demo-phone-wrap">
            <div className="demo-channel-switch" aria-label="Choose a messaging channel">
              <button type="button" className={chatChannel === 'whatsapp' ? 'active whatsapp' : ''} aria-pressed={chatChannel === 'whatsapp'} onClick={() => selectChannel('whatsapp')}><MessageCircle size={15} /> WhatsApp</button>
              <button type="button" className={chatChannel === 'telegram' ? 'active telegram' : ''} aria-pressed={chatChannel === 'telegram'} onClick={() => selectChannel('telegram')}><Send size={15} /> Telegram</button>
            </div>

            <div className={`demo-phone demo-phone-${chatChannel}`}>
              <div className="demo-phone-bar"><i /><span /><i /></div>
              <div className="demo-app-head">
                <div className="demo-avatar">N</div>
                <div><strong>NourishBot</strong><small>{chatChannel === 'whatsapp' ? 'WhatsApp · preview' : 'Telegram bot · preview'}</small></div>
                <span>{chatChannel === 'whatsapp' ? <MessageCircle size={17} /> : <Send size={17} />}</span>
              </div>

              <div className="demo-messages" ref={messagesRef} aria-live="polite" aria-label="Illustrative conversation preview" tabIndex={0}>
                <div className="demo-day-label">Illustrative conversation</div>
                <div className="demo-bubble demo-bubble-user">{conversation.first}<small>12:02 {chatChannel === 'whatsapp' ? '✓✓' : '✓'}</small></div>
                <div className="demo-bubble demo-bubble-bot">{conversation.reply}</div>

                {chatFeature === 'meals' ? (
                  <div className="demo-product-card demo-meal-plan" aria-label="Editable illustrative meal plan">
                    <header>
                      <div className="demo-summary-icon"><Home size={16} /></div>
                      <div><small>TOMORROW’S PLAN</small><strong>Four meals at home</strong></div>
                      <span>Draft</span>
                    </header>
                    <div className="demo-plan-totals">
                      <span><small>EXAMPLE COST</small><strong>₹{mealTotals.price} of ₹800</strong></span>
                      <span><small>EXAMPLE NUTRITION</small><strong>{mealTotals.kcal.toLocaleString('en-IN')} kcal · {mealTotals.protein}g protein</strong></span>
                    </div>
                    <div className="demo-meal-slots">
                      {mealSlots.map((meal, index) => {
                        const selected = meal.options[mealSelections[index]]
                        return (
                          <button type="button" key={meal.slot} onClick={() => cycleMeal(index)} aria-label={`Change ${meal.slot} from ${selected.name}`}>
                            <time>{meal.time}</time>
                            <span><small>{meal.slot}</small><strong>{selected.name}</strong><em>{selected.kcal} kcal · {selected.protein}g protein</em></span>
                            <i>Change</i>
                          </button>
                        )
                      })}
                    </div>
                    <p className="demo-data-note">Example menu values only. Real nutrition needs complete, source-labelled data.</p>
                  </div>
                ) : (
                  <div className="demo-product-card demo-route-preview" aria-label="Illustrative route pickup options">
                    <div className="demo-mini-map" role="img" aria-label="Illustrative route from Indiranagar to Whitefield with a food stop">
                      <span className="map-origin">A</span><i className="map-road" /><b><Navigation size={13} /></b><i className="map-road map-road-last" /><span className="map-destination">B</span>
                      <small>Indiranagar</small><small>Whitefield</small>
                    </div>
                    <header>
                      <div className="demo-summary-icon"><Navigation size={16} /></div>
                      <div><small>ON YOUR ROUTE</small><strong>Two pickup candidates</strong></div>
                      <span>Demo</span>
                    </header>
                    <div className="demo-route-candidates">
                      {routeCandidates.map((candidate, index) => (
                        <button type="button" key={candidate.name} className={routeCandidate === index ? 'active' : ''} aria-pressed={routeCandidate === index} onClick={() => { setRouteCandidate(index); setPreviewReviewed(false) }}>
                          <span><strong>{candidate.name}</strong><small>{candidate.food} · {candidate.area}</small><small>Estimated detour {candidate.detour}</small></span>
                          <i>{routeCandidate === index ? 'Selected' : 'Choose'}</i>
                        </button>
                      ))}
                    </div>
                    <div className="demo-pickup-details">
                      <span><small>ROUTE ARRIVAL</small><strong>{activeRoute.arrival}</strong></span>
                      <span><small>FUTURE PICKUP</small><strong>{activeRoute.pickup}</strong></span>
                      <span><small>PROVIDER STATUS</small><strong>{activeRoute.status}</strong></span>
                    </div>
                    <p className="demo-data-note">Example travel times only. Arrival does not guarantee food readiness.</p>
                  </div>
                )}

                {previewReviewed && (
                  <>
                    <div className="demo-bubble demo-bubble-bot demo-bubble-later">{conversation.final}</div>
                    <div className="demo-safety-note"><ShieldCheck size={14} /><span><strong>Confirmation stays separate.</strong> {chatFeature === 'meals' ? 'A meal plan does not place an order.' : 'A route choice does not place an order or promise readiness.'}</span></div>
                  </>
                )}
              </div>

              <div className="demo-review-bar">
                <button className="demo-chat-action" type="button" onClick={() => setPreviewReviewed((current) => !current)} aria-expanded={previewReviewed}>
                  {previewReviewed ? 'Back to preview' : chatFeature === 'meals' ? 'Review meal plan' : 'Review pickup details'} <ArrowRight size={15} />
                </button>
              </div>
              <div className="demo-compose"><span>{chatChannel === 'whatsapp' ? 'Message' : 'Write a message…'}</span><button type="button" disabled aria-label={`Sending is disabled in this ${chatChannel} demo`}><Send size={15} /></button></div>
            </div>
            <p className="demo-caption">Interactive illustration · No live WhatsApp, Telegram, Maps or provider connection</p>
          </div>
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
        <nav aria-label="Footer navigation"><a href="#how-it-works">How it works</a><a href="#meal-planning">Home meals</a><a href="#route-pickup">Route pickup</a><a href="#chat-demo">Chat demo</a><Link href="/privacy">Privacy</Link><Link href="/login">Sign in</Link></nav>
        <p>Discovery information is verified through connected services where available. A plan or provider handoff is not a placed order until you explicitly confirm it.</p>
        <span className="ed-footer-copy">© 2026 NourishBot</span>
      </footer>
    </>
  )
}
