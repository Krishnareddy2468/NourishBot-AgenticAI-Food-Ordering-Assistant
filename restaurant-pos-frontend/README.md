# Dzukku Frontend

Restaurant POS and operations dashboard for Dzukku.

## What It Does

- Loads menu, analytics, tables, reservations, employees, and invoices from `public/Project_Dzukku.xlsx`
- Connects live POS order creation and status updates to the backend `/v1` APIs
- Shows payment intent and settlement breakdowns in the invoice flow

## Run

```bash
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `.env` if your backend is not running on `http://localhost:8000`.

## Verify

```bash
npm run lint
npm run build
```

## Notes

- Historical data is still Excel-driven.
- Newly created POS orders are backend-driven and merged into the dashboard.
- WhatsApp webhook handling is intentionally not covered in the frontend.
