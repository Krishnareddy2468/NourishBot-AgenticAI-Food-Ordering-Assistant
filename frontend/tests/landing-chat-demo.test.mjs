import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const landing = await readFile(new URL('../components/landing/EditorialSections.jsx', import.meta.url), 'utf8')
const home = await readFile(new URL('../app/page.jsx', import.meta.url), 'utf8')

test('landing page presents the two Phase 9 customer experiences', () => {
  assert.match(landing, /NUTRITION-AWARE HOME MEALS/)
  assert.match(landing, /ON-ROUTE PICKUP/)
  assert.match(home, /Eat well at home/)
  assert.match(home, /Pick up on the way/)
  assert.match(landing, /chatFeature === key \? 'Selected' : 'Explore'/)
})

test('chat demo switches between WhatsApp and Telegram', () => {
  assert.match(landing, /chatChannel === 'whatsapp'/)
  assert.match(landing, /channelConversations/)
  assert.match(landing, /selectChannel\('whatsapp'\)/)
  assert.match(landing, /selectChannel\('telegram'\)/)
  assert.match(landing, /> WhatsApp</)
  assert.match(landing, /> Telegram</)
  assert.match(landing, /aria-pressed={chatChannel === 'whatsapp'}/)
  assert.match(landing, /aria-pressed={chatChannel === 'telegram'}/)
})

test('demo preserves the product confirmation and verification boundaries', () => {
  assert.match(landing, /no meal has been ordered/i)
  assert.match(landing, /explicit confirmation/i)
  assert.match(landing, /does not guarantee food readiness/i)
  assert.match(landing, /nutrition needs complete, source-labelled data/i)
  assert.match(landing, /not a live order/i)
})

test('meal preview contains editable slots, budget and illustrative nutrition context', () => {
  assert.match(landing, /Breakfast/)
  assert.match(landing, /Lunch/)
  assert.match(landing, /Snack/)
  assert.match(landing, /Dinner/)
  assert.match(landing, /mealTotals/)
  assert.match(landing, /cycleMeal\(index\)/)
  assert.match(landing, /Example menu values only/)
})

test('route preview contains candidates, detours, future pickup and provider status', () => {
  assert.match(landing, /routeCandidates/)
  assert.match(landing, /estimated detour/i)
  assert.match(landing, /FUTURE PICKUP/)
  assert.match(landing, /PROVIDER STATUS/)
  assert.match(landing, /Vegetarian grain bowls/)
  assert.match(landing, /role="img"/)
  assert.match(landing, /setRouteCandidate\(index\)/)
})

test('review action stays outside the scrolling transcript and above the composer', () => {
  const transcript = landing.indexOf('className="demo-messages"')
  const transcriptEnd = landing.indexOf('className="demo-review-bar"')
  const action = landing.indexOf('className="demo-chat-action"')
  const composer = landing.indexOf('className="demo-compose"')
  assert.ok(transcript >= 0 && transcriptEnd > transcript)
  assert.ok(action > transcriptEnd && composer > action)
})
