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
