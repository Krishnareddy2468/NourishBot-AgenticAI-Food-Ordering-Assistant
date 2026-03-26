# Zomatobot Frontend

Next.js web chat UI for the Zomatobot backend.

## Local Setup

1. Install dependencies:

```bash
npm install
```

2. Create env file:

```bash
cp .env.example .env.local
```

3. Start backend first (from `backend/`):

```bash
python3 main.py
```

4. Start frontend:

```bash
npm run dev
```

5. Open http://localhost:3000

## Backend Connectivity

The frontend now uses internal Next.js API routes:

- `POST /api/zomato/chat` -> forwards to backend `POST /api/chat/message`
- `POST /api/zomato/voice-upload` -> forwards to backend `POST /api/chat/voice-upload`
- `GET /api/zomato/order-status/:userId` -> forwards to backend `GET /api/chat/order-status/:userId`

This avoids browser CORS issues and keeps backend URL configuration server-side.

## Environment Variables

- `BACKEND_API_URL` (preferred): URL for server-side forwarding, e.g. `http://localhost:8000`
- `NEXT_PUBLIC_API_URL` (optional): fallback URL used if `BACKEND_API_URL` is not set

## Notes

- Ensure `ENABLE_WEB_CHAT=true` in `backend/.env` so chat routes are enabled.
- If backend is unavailable, the frontend returns a friendly connection error in chat.
- Voice input is available from the mic button near the message input.
