# DzukkuBot Codebase Guide

This document explains what this project does, how messages move through the system, where data is stored, and what each file is responsible for.

## 1) What this project is

DzukkuBot is a restaurant assistant chatbot for two channels:
- Telegram
- WhatsApp (via Twilio webhook)

The bot can:
- answer menu questions
- collect and place food orders
- collect and create table reservations
- answer restaurant info questions (timings, delivery, etc.)

It uses:
- Groq LLM for chat intelligence
- SQLite for local storage
- Google Sheets for order and reservation sync

## 2) High-level architecture

Message flow:
1. User sends message in Telegram or WhatsApp.
2. Channel handler receives message.
3. Message goes to bot_brain.py.
4. bot_brain.py builds prompt + menu context from SQLite.
5. Groq LLM returns response.
6. If response contains an order/reservation tag, data is parsed and saved:
   - local SQLite DB
   - Google Sheets
7. Clean reply is sent back to user.

Storage flow:
- Menu source starts in data/restaurant_dataset.csv
- database_setup.py imports CSV into database/dzukku.db table menu
- orders/reservations are appended by bot_brain.py through database_setup.py
- sheets_sync.py mirrors orders/reservations into Google Sheets tabs

## 3) File-by-file explanation

### main.py
Primary runtime entrypoint for deployment.
- Starts Telegram bot in a background thread.
- Starts WhatsApp Flask app in main thread.
- Uses TELEGRAM_TOKEN and PORT from environment.

### telegram_bot.py
Telegram channel implementation.
- Defines /start, /menu, /help commands.
- Converts keyboard button text into plain intent text.
- Maintains per-user in-memory conversation history.
- Calls get_bot_response from bot_brain.py.
- Sends response back to Telegram chat.

### whatsapp_bot.py
WhatsApp channel implementation.
- Flask route: POST /whatsapp
- Reads incoming Body and From from webhook payload.
- Maintains per-sender in-memory history.
- Calls get_bot_response from bot_brain.py.
- Returns Twilio MessagingResponse.

### bot_brain.py
Core intelligence and transaction extraction.
- Builds system prompt with menu text from DB.
- Calls Groq chat completion API.
- Looks for special structured tags in model response:
  - ##ORDER##name|phone|items|total_price##
  - ##RESERVATION##name|phone|date|time|guests|special_request##
- Parses tag data and saves to DB + Google Sheets.
- Removes tag text before sending message to user.

### database_setup.py
Database schema and persistence logic.
- Creates tables:
  - menu
  - customers
  - orders
  - reservations
- Imports CSV menu into menu table.
- Saves new orders and reservations.
- Formats menu text for prompt use by bot_brain.py.

### sheets_sync.py
Google Sheets integration.
- Auth via GOOGLE_CREDENTIALS env JSON or credentials.json file.
- Opens sheet by GOOGLE_SHEET_ID.
- Appends order rows to worksheet Orders.
- Appends reservation rows to worksheet Reservations.

### data/restaurant_dataset.csv
Menu seed data used to populate SQLite menu table.

### database/dzukku.db
SQLite database file created and used at runtime.

### run_all.py
Alternative launcher that starts telegram_bot.py and whatsapp_bot.py as separate subprocesses in threads.

### generate_qr.py
Utility script that generates QR codes for:
- Telegram bot link
- WhatsApp chat link

### Procfile
Deployment process definition:
- web: python main.py

### railway.toml and nixpacks.toml
Railway and Nixpacks deployment configuration.

### requirements.txt
Python dependencies required by this project.

### start.bat
Windows helper script to start Telegram bot, WhatsApp bot, and ngrok in separate command windows.

### dzukku_data.json
Legacy/static restaurant JSON sample data. Not the primary source used by runtime bot flow.

## 4) Environment variables required

Create a .env file with values like:

- GROQ_API_KEY=your_groq_api_key
- TELEGRAM_TOKEN=your_telegram_bot_token
- GOOGLE_SHEET_ID=your_google_sheet_id
- GOOGLE_CREDENTIALS={...service_account_json...} (optional if credentials.json exists)
- PORT=8080 (optional)

## 5) How to run locally

1. Install dependencies:
   pip install -r requirements.txt

2. Initialize DB and import menu:
   python database_setup.py

3. Start both bots:
   python main.py

## 6) Important runtime notes

- Conversation history is in-memory per user/sender, so it resets when process restarts.
- Orders and reservations are persisted in SQLite and synced to Google Sheets.
- If Groq call fails, user gets a graceful fallback message.
- The bot behavior is strongly controlled through the prompt in bot_brain.py.

## 7) Common troubleshooting

### Bot does not respond
- Check TELEGRAM_TOKEN / Twilio webhook setup.
- Check process logs for exceptions.

### LLM errors
- Verify GROQ_API_KEY is valid and has quota.

### Google Sheets sync fails
- Verify GOOGLE_SHEET_ID.
- Verify service account permissions on the sheet.
- Confirm worksheet names exist exactly: Orders and Reservations.

### Menu looks wrong
- Recheck CSV columns and run database_setup.py again.
- Confirm menu table contains updated rows in database/dzukku.db.

## 8) Suggested next improvements

- Add validation for phone/date/time formats before saving.
- Add retry and dead-letter logging for Sheets sync failures.
- Add unit tests for extract_and_save parser logic.
- Move in-memory history to Redis/DB for multi-instance deployments.
- Add admin dashboard for order and reservation management.
