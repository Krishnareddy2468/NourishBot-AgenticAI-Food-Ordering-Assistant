"""
Telegram Bot — NourishBot Restaurant
=================================
Agentic bot using OpenAI GPT-4o via agent_orchestrator.
Features:
  - Persistent session state per user (PostgreSQL)
  - Inline keyboards for quick actions
  - Canteen menu, cart, scheduled collection, and home-delivery assistance
  - Typing indicator while thinking
  - /start, /menu, /order, /pickup, /cart, /help, /reset
"""

import logging
import os
import sys

# Global bot instance for cross-module access (notifications)
_bot_instance = None


def get_bot_instance():
    """Return the running bot instance for sending notifications."""
    return _bot_instance

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv

from app.agent.pipeline import process_message as pipeline_process
from app.core.config import settings
from app.db.crud import get_session, reset_session, save_session, save_order_rating

load_dotenv()
logger = logging.getLogger(__name__)

# Fix Windows console emoji output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ── Keyboards ─────────────────────────────────────────────────────────────────

def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📋 Menu"), KeyboardButton("⭐ Specials")],
            [KeyboardButton("🗓️ Schedule Meal"), KeyboardButton("🚗 Find on my route")],
            [KeyboardButton("🏠 Home Delivery"), KeyboardButton("🛍️ My Cart")],
            [KeyboardButton("🔌 Choose Provider"), KeyboardButton("ℹ️ Info")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Type a message or pick an option…",
    )


def quick_actions_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 View Menu",      callback_data="action_menu"),
            InlineKeyboardButton("⭐ Today's Specials", callback_data="action_specials"),
        ],
        [
            InlineKeyboardButton("🗓️ Schedule a Meal", callback_data="action_order"),
            InlineKeyboardButton("🚗 Find on my route", callback_data="action_drive_thru"),
        ],
        [
            InlineKeyboardButton("🏠 Home Delivery", callback_data="action_delivery"),
            InlineKeyboardButton("🛍️ View My Cart", callback_data="action_cart"),
        ],
        [
            InlineKeyboardButton("🔌 Choose Provider",  callback_data="action_providers"),
            InlineKeyboardButton("ℹ️ Service Info",     callback_data="action_info"),
        ],
    ])


def provider_inline() -> InlineKeyboardMarkup:
    providers = ["🍽️ NourishBot", "🟠 Swiggy", "🔴 Zomato"]
    buttons = []
    for display in providers:
        name = display.split(" ", 1)[1]
        if len(f"provider_{name}".encode("utf-8")) <= 64:
            buttons.append(InlineKeyboardButton(display, callback_data=f"provider_{name}"))
    return InlineKeyboardMarkup([buttons[i:i + 2] for i in range(0, len(buttons), 2)])


# ── Rating keyboard ───────────────────────────────────────────────────────────

def rating_inline_keyboard(order_ref: str) -> InlineKeyboardMarkup:
    """Inline keyboard for post-order rating (1–5 stars)."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⭐",       callback_data=f"rating_1_{order_ref}"),
            InlineKeyboardButton("⭐⭐",      callback_data=f"rating_2_{order_ref}"),
            InlineKeyboardButton("⭐⭐⭐",     callback_data=f"rating_3_{order_ref}"),
            InlineKeyboardButton("⭐⭐⭐⭐",   callback_data=f"rating_4_{order_ref}"),
            InlineKeyboardButton("⭐⭐⭐⭐⭐",  callback_data=f"rating_5_{order_ref}"),
        ]
    ])


# Greeting detection
GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "yo",
    "namaste", "namaskar", "namaskaram", "vanakkam",
    "hola", "salaam", "salam", "good morning", "good afternoon",
    "good evening", "gm", "gn", "ge", "ga", "start",
}


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower().rstrip("!.?,…")
    if not t:
        return False
    return t in GREETING_WORDS or t.split()[0] in GREETING_WORDS


# ── Button text → natural language intent ────────────────────────────────────

BUTTON_INTENT_MAP = {
    "📋 Menu":            "Show me the full menu please",
    "⭐ Specials":         "What are today's specials and deals?",
    "🗓️ Schedule Meal":  "I want to schedule a meal for a future date and time",
    "🚗 Find on my route": "I want a scheduled drive-through pickup. Ask for my route origin and destination, then use Maps to find available restaurant candidates along the route.",
    "🏠 Home Delivery":   "I want to schedule a meal for home delivery at a future date and time",
    "🛍️ My Cart":        "Show me my current cart",
    "🔌 Choose Provider": "Show me the available meal providers",
    "ℹ️ Info":            "Explain scheduled meal ordering and the supported channels",
}

CALLBACK_INTENT_MAP = {
    "action_menu":     "Show me the full menu please",
    "action_specials": "What are today's specials and deals?",
    "action_order":    "I want to schedule a meal for a future date and time",
    "action_drive_thru":"I want a scheduled drive-through pickup. Ask for my route origin and destination, then use Maps to find available restaurant candidates along the route.",
    "action_delivery":  "I want to schedule a meal for home delivery at a future date and time",
    "action_cart":     "Show me my current cart",
    "action_providers":"Show me the available meal providers",
    "action_info":     "Explain scheduled meal ordering and the supported channels",
}


# ── Helper: send typing + call agent ──────────────────────────────────────────

async def _think_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str):
    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else ""

    try:
        await _do_think_and_reply(update, context, chat_id, user_name, user_message)
    except Exception as e:
        logger.exception("_think_and_reply crashed (chat=%s): %s", chat_id, e)
        try:
            await update.effective_message.reply_text(
                "I hit a snag — could you try that again?",
                reply_markup=main_keyboard(),
            )
        except Exception:
            pass  # best effort


async def _do_think_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_name: str,
    user_message: str,
) -> None:
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    session = await get_session(chat_id)
    platform = session.get("ordering_platform") or "NourishBot"

    try:
        if platform in ("NourishBot", ""):
            reply = await pipeline_process(
                message=user_message,
                chat_id=chat_id,
                user_name=user_name,
                channel_type="TELEGRAM",
            )
        else:
            from app.agent.mcp_agent import get_mcp_response
            reply = await get_mcp_response(user_message, chat_id, user_name, platform)
            if reply is None:
                reply = (
                    f"{platform} scheduled ordering is unavailable right now. "
                    "Choose NourishBot, Swiggy, or Zomato; I will not place an instant order."
                )
    except Exception as e:
        logger.error("Canteen pipeline failed: %s", e, exc_info=True)
        reply = None

    if reply is None:
        reply = (
            "Sorry, I had trouble reaching the meal service for a moment. "
            "Please try again — I need the live menu/cart DB before I can order safely."
        )

    try:
        await update.effective_message.reply_text(
            reply,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_keyboard(),
        )
    except Exception:
        # Fall back to plain text when markdown is malformed
        await update.effective_message.reply_text(
            reply,
            reply_markup=main_keyboard(),
        )


# ── /start ────────────────────────────────────────────────────────────────────

async def _send_quick_actions(update: Update):
    """Show the active scheduled-meal actions."""
    await update.effective_message.reply_text(
        "What would you like to do? Choose an option or describe your meal naturally.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=quick_actions_inline(),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else "there"

    await reset_session(chat_id, user_name)
    welcome = (
        f"👋 Hey *{user_name}*! Welcome to *NourishBot* 🍽️\n\n"
        f"Plan breakfast, lunch, snacks, or dinner for pickup, drive-through, or home delivery. "
        f"Instant ordering is disabled—every meal needs a future date and time."
    )
    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=provider_inline(),
    )
    await _send_quick_actions(update)


# ── /menu ──────────────────────────────────────────────────────────────────────

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "Show me the complete menu with prices")


# ── /order ─────────────────────────────────────────────────────────────────────

async def cmd_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "I want to schedule a meal for a future date and time")


# ── /pickup ────────────────────────────────────────────────────────────────────

async def cmd_pickup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "Explain NourishBot pickup and external provider fulfillment")


# ── /cart ──────────────────────────────────────────────────────────────────────

async def cmd_cart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "Show me my current cart")


# ── /reset ─────────────────────────────────────────────────────────────────────

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else ""
    await reset_session(chat_id, user_name)
    await update.message.reply_text(
        "✅ Session reset! Starting fresh 🍽️\nWhat can I get you?",
        reply_markup=main_keyboard(),
    )


# ── /help ──────────────────────────────────────────────────────────────────────

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *What I can do for you:*\n\n"
        "📋 /menu — Browse the full menu\n"
        "🛒 /order — Schedule a future meal\n"
        "📦 /pickup — Learn how fulfillment works\n"
        "🛍️ /cart — View your cart\n"
        "🔄 /reset — Start a fresh conversation\n"
        "❓ /help — Show this message\n\n"
        "Or just *chat naturally* — I'll understand! 😊\n"
        "Try: _\"Find vegetarian drive-through restaurants from Madhapur to Hyderabad Airport, then schedule one for tomorrow at 1 PM IST\"_\n\n"
        "Every order needs a future time. Instant ordering is disabled."
    )
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_keyboard(),
    )


# ── Inline callback handler ───────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else ""

    # ── Rating callback (Phase 1) ───────────────────────────────────────────
    if query.data and query.data.startswith("rating_"):
        parts = query.data.split("_", 2)
        if len(parts) == 3:
            rating = int(parts[1])
            order_ref = parts[2]
            ok = await save_order_rating(order_ref, rating)
            if ok:
                try:
                    from sqlalchemy import select as _sel
                    from app.db.session import AsyncSessionLocal
                    from app.db.models import Order as OrderModel, OrderItem
                    from app.agent.memory_agent import update_taste_vector
                    async with AsyncSessionLocal() as _db:
                        _or = (await _db.execute(
                            _sel(OrderModel).where(OrderModel.order_ref == order_ref)
                        )).scalar_one_or_none()
                        if _or and _or.customer_id:
                            _ir = await _db.execute(
                                _sel(OrderItem).where(OrderItem.order_id == _or.id)
                            )
                            items = [
                                {"item_name": oi.item_name_snapshot, "qty": oi.qty,
                                 "price_cents": oi.unit_price_cents}
                                for oi in _ir.scalars().all()
                            ]
                            import asyncio as _aio
                            _aio.create_task(update_taste_vector(_or.customer_id, items, rating=rating))
                except Exception as _e:
                    logger.debug("Rating feedback skipped: %s", _e)
                stars = "⭐" * rating
                await query.edit_message_text(
                    f"Thanks! You rated this order {stars}\nYour preferences have been updated.",
                    reply_markup=None,
                )
            else:
                await query.edit_message_text(
                    "Sorry, I couldn't save your rating. The order may no longer exist.",
                    reply_markup=None,
                )
        return

    intent = CALLBACK_INTENT_MAP.get(query.data)
    if query.data and query.data.startswith("provider_"):
        platform = query.data.removeprefix("provider_")
        await save_session(chat_id, {"ordering_platform": platform})
        await query.edit_message_text(
            f"✅ *{platform}* selected. Tell me the meal, future date, and time you want to schedule.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    if query.data == "action_providers":
        await query.message.reply_text("Choose a scheduled-meal provider:", reply_markup=provider_inline())
        return
    if not intent:
        return

    await _think_and_reply(update, context, intent)


# ── Text message handler ──────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text or ""

    if raw == "🔌 Choose Provider":
        await update.message.reply_text("Choose a scheduled-meal provider:", reply_markup=provider_inline())
        return

    # Offer quick actions for a fresh greeting without interrupting an active cart.
    if _is_greeting(raw) and raw not in BUTTON_INTENT_MAP:
        chat_id = update.effective_chat.id
        sess    = await get_session(chat_id)
        cart    = sess.get("cart", []) or []
        if not cart:
            user      = update.effective_user
            user_name = (user.first_name if user else "") or sess.get("user_name") or "there"

            greeting_line = f"👋 Hey *{user_name}*! Welcome to *NourishBot* 🍽️"
            await update.message.reply_text(
                greeting_line,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_keyboard(),
            )
            await _send_quick_actions(update)
            return

    # Map keyboard buttons to natural language
    message = BUTTON_INTENT_MAP.get(raw, raw)
    await _think_and_reply(update, context, message)


# ── Location message handler ──────────────────────────────────────────────────

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    location = update.message.location
    await _think_and_reply(
        update,
        context,
        (
            f"My current location is {location.latitude:.6f},{location.longitude:.6f}. "
            "Use it as my route origin. If I want drive-through, ask for my destination; "
            "if I want home delivery, ask for the complete delivery address."
        ),
    )


# ── Error handler ──────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Telegram error: %s", context.error, exc_info=context.error)


# ── App builder ───────────────────────────────────────────────────────────────

def build_app():
    global _bot_instance
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is not set in .env")

    app = ApplicationBuilder().token(token).build()
    _bot_instance = app.bot

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("menu",    cmd_menu))
    app.add_handler(CommandHandler("order",   cmd_order))
    app.add_handler(CommandHandler("pickup",  cmd_pickup))
    app.add_handler(CommandHandler("cart",    cmd_cart))
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("help",    cmd_help))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.add_error_handler(error_handler)
    return app
