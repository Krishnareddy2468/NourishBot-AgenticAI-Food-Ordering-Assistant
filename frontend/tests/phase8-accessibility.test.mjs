import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const account = await readFile(new URL('../app/account/page.jsx', import.meta.url), 'utf8')
const analytics = await readFile(new URL('../components/pages/views/AnalyticsView.jsx', import.meta.url), 'utf8')
const layout = await readFile(new URL('../app/layout.jsx', import.meta.url), 'utf8')
const config = await readFile(new URL('../next.config.mjs', import.meta.url), 'utf8')

test('customer dashboard has semantic landmarks and named interactive controls', () => {
  assert.match(account, /<main className=/)
  assert.match(account, /<header className=/)
  assert.match(account, /<nav className=.*aria-label=/)
  assert.match(account, /aria-labelledby="week-title"/)
  assert.match(account, /aria-label="Previous week"/)
  assert.match(account, /aria-label="Next week"/)
  assert.match(account, /role="alert"/)
  assert.match(account, /aria-pressed=/)
})

test('forms use explicit labels and status changes are announced', () => {
  assert.match(account, /<label>Dietary choices/)
  assert.match(account, /<label>Allergies/)
  assert.match(account, /<label>Health goals/)
  assert.match(account, /role="status"/)
  assert.match(account, /aria-label=\{`New time for/)
})

test('staff analytics has bounded date controls and visible empty states', () => {
  assert.match(analytics, /type="date"/)
  assert.match(analytics, /No scheduled meals in this range/)
  assert.match(analytics, /No completed revenue in this range/)
  assert.match(analytics, /disabled={loading}/)
})

test('site metadata and browser security policies are configured', () => {
  assert.match(layout, /metadata/)
  assert.match(config, /Content-Security-Policy/)
  assert.match(config, /frame-ancestors 'none'/)
  assert.match(config, /X-Content-Type-Options/)
})

test('customer brand uses the custom NourishBot logo rather than prohibited marks', () => {
  assert.match(account, /NourishLogo/)
  assert.doesNotMatch(account, /\bLeaf\b/)
  assert.doesNotMatch(account, /Sparkles?/)
})
