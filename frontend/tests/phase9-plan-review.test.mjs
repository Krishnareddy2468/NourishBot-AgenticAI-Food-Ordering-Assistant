import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const account = await readFile(new URL('../app/account/page.jsx', import.meta.url), 'utf8')
const platformApi = await readFile(new URL('../services/platformApi.js', import.meta.url), 'utf8')

test('customers can review and edit draft plan meals without placing orders', () => {
  assert.match(account, /Review \/ edit/)
  assert.match(account, /Save meal/)
  assert.match(account, /updateCustomerPlannedMeal/)
  assert.match(account, /Meal plan confirmed\. No order was placed\./)
  assert.match(platformApi, /export async function fetchCustomerMealMenu/)
  assert.match(platformApi, /export async function updateCustomerPlannedMeal/)
})

test('confirmed future meals expose the safe chat order handoff command', () => {
  assert.match(account, /meal\.state === 'CONFIRMED'/)
  assert.match(account, /meal\.order_id/)
  assert.match(account, /order planned meal #\$\{meal\.id\}/)
  assert.match(account, /Copy order command/)
  assert.match(account, /Nothing has been ordered\./)
})

test('confirmed future meals can be ordered from the dashboard in two explicit steps', () => {
  // Step one reviews; only a separate confirmation places the order.
  assert.match(account, /Order this meal/)
  assert.match(account, /Review this order/)
  assert.match(account, /Confirm and place order/)
  assert.match(account, /previewPlannedMealOrder/)
  assert.match(account, /confirmPlannedMealOrder/)
  assert.match(account, /confirmation_id: checkout\.preview\.confirmation_id/)
  assert.match(platformApi, /export async function previewPlannedMealOrder/)
  assert.match(platformApi, /export async function confirmPlannedMealOrder/)
  assert.match(platformApi, /planned-meals\/\$\{mealId\}\/order\/preview/)
  assert.match(platformApi, /planned-meals\/\$\{mealId\}\/order\/confirm/)
})

test('a changed review is discarded rather than silently confirmed', () => {
  // Editing any checkout field clears the stored preview, so the customer
  // cannot confirm a summary that no longer matches what they chose.
  assert.match(account, /orderType: event\.target\.value, preview: null/)
  assert.match(account, /deliveryAddress: event\.target\.value, preview: null/)
  assert.match(account, /contactPhone: event\.target\.value, preview: null/)
})
