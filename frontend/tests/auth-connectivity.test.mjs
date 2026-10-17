import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const apiClient = await readFile(new URL('../services/platformApi.js', import.meta.url), 'utf8')
const loginRoute = await readFile(new URL('../app/login/page.jsx', import.meta.url), 'utf8')
const loginPage = await readFile(new URL('../components/pages/LoginPage.jsx', import.meta.url), 'utf8')

test('browser API client preserves credentials and normalizes loopback hosts', () => {
  assert.match(apiClient, /credentials: 'include'/)
  assert.match(apiClient, /loopbackHosts/)
  assert.match(apiClient, /url\.hostname = window\.location\.hostname/)
})

test('network failures identify the backend URL and configuration variable', () => {
  assert.match(apiClient, /Cannot reach the NourishBot API/)
  assert.match(apiClient, /NEXT_PUBLIC_API_BASE_URL/)
  assert.match(loginPage, /setError\(err\.message/)
})

test('authenticated customers go directly to their account dashboard', () => {
  assert.match(loginRoute, /role === 'CUSTOMER' \? '\/account' : '\/workspace'/)
})
