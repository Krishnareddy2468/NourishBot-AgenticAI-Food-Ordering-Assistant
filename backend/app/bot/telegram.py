"""
Telegram Bot — Dzukku Restaurant
=================================
Agentic bot using OpenAI GPT-4o via agent_orchestrator.
Features:
  - Persistent session state per user (PostgreSQL)
  - Inline keyboards for quick actions
  - Location sharing → nearby restaurant search hook
  - Typing indicator while thinking
  - /start, /menu, /order, /reserve, /cart, /help, /reset
"""

import asyncio
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

from app.agent.orchestrator import get_bot_response
from app.agent.mcp_agent import get_mcp_response
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
            [KeyboardButton("🛒 Order"), KeyboardButton("📅 Reserve a Table")],
            [KeyboardButton("🛍️ My Cart"), KeyboardButton("ℹ️ Info")],
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
            InlineKeyboardButton("🛒 Place an Order",   callback_data="action_order"),
            InlineKeyboardButton("📅 Book a Table",     callback_data="action_reserve"),
        ],
        [
            InlineKeyboardButton("🛍️ View My Cart",    callback_data="action_cart"),
            InlineKeyboardButton("ℹ️ Restaurant Info", callback_data="action_info"),
        ],
    ])


# ── Platform selection (Dzukku / Zomato / Swiggy) ─────────────────────────────

def platform_selection_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🍽️ Order via Dzukku Bot", callback_data="platform_dzukku")],
        [
            InlineKeyboardButton("🟥 Zomato",  callback_data="platform_zomato"),
            InlineKeyboardButton("🟧 Swiggy",  callback_data="platform_swiggy"),
        ],
    ])


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


PLATFORM_PROMPT_TEXT = (
    "Where would you like to order from? 🍽️\n\n"
    "• *Dzukku Bot* — chat & order right here\n"
    "• *Zomato* / *Swiggy* — order via the delivery app"
)


# Greeting detection (used to re-show platform prompt on hello/hi/etc.)
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
    "🛒 Order":           "I want to place an order",
    "📅 Reserve a Table": "I want to reserve a table",
    "🛍️ My Cart":        "Show me my current cart",
    "ℹ️ Info":            "Tell me about the restaurant — timings, location, delivery",
}

CALLBACK_INTENT_MAP = {
    "action_menu":     "Show me the full menu please",
    "action_specials": "What are today's specials and deals?",
    "action_order":    "I want to place an order",
    "action_reserve":  "I want to reserve a table",
    "action_cart":     "Show me my current cart",
    "action_info":     "Tell me about the restaurant — timings, location, delivery",
}


PLATFORM_CALLBACKS = {
    "platform_dzukku": "Dzukku",
    "platform_zomato": "Zomato",
    "platform_swiggy": "Swiggy",
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

    # Routing matrix:
    #   ordering_platform = Zomato/Swiggy + MCP_ENABLED → mcp_agent (LangGraph + MCP)
    #   ordering_platform = Dzukku (or unset)           → deterministic DB pipeline
    #   external MCP unavailable                        → legacy link fallback
    reply: str | None = None
    sess = await get_session(chat_id)
    platform_choice = (sess.get("ordering_platform") or "").strip()

    if platform_choice in ("Zomato", "Swiggy") and settings.MCP_ENABLED:
        try:
            reply = await get_mcp_response(
                user_message=user_message,
                chat_id=chat_id,
                user_name=user_name,
                platform=platform_choice,
            )
        except Exception as e:
            logger.error("MCP agent failed, falling back: %s", e, exc_info=True)
            reply = None

    if reply is None and platform_choice not in ("Zomato", "Swiggy"):
        # In-house Dzukku flow: 5-stage deterministic pipeline.
        try:
            reply = await pipeline_process(
                message=user_message,
                chat_id=chat_id,
                user_name=user_name,
            )
        except Exception as e:
            logger.error("Pipeline failed, falling back to legacy: %s", e, exc_info=True)
            reply = None

    if reply is None and platform_choice in ("Zomato", "Swiggy"):
        # Last-resort fallback for external ordering: provide links / legacy
        # behaviour if live MCP is unavailable.
        reply = await asyncio.get_event_loop().run_in_executor(
            None,
            get_bot_response,
            user_message,
            chat_id,
            user_name,
        )

    if reply is None:
        reply = (
            "Sorry, I had trouble reaching the restaurant system for a moment. "
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

async def _send_platform_prompt(update: Update):
    """Send the Dzukku / Zomato / Swiggy platform-selection prompt."""
    await update.effective_message.reply_text(
        PLATFORM_PROMPT_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=platform_selection_inline(),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else "there"

    await reset_session(chat_id, user_name)

    welcome = (
        f"👋 Hey *{user_name}*! Welcome to *Dzukku Restaurant* 🍽️\n\n"
        f"_Where every bite hits different ❤️_\n\n"
        f"I'm your AI-powered restaurant assistant — I can help you with "
        f"our menu, take your order, or book a table."
    )
    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_keyboard(),
    )
    await _send_platform_prompt(update)


# ── /menu ──────────────────────────────────────────────────────────────────────

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "Show me the complete menu with prices")


# ── /order ─────────────────────────────────────────────────────────────────────

async def cmd_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "I want to place an order")


# ── /reserve ───────────────────────────────────────────────────────────────────

async def cmd_reserve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _think_and_reply(update, context, "I want to reserve a table")


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
        "🛒 /order — Place a food order\n"
        "📅 /reserve — Book a table\n"
        "🛍️ /cart — View your cart\n"
        "🔄 /reset — Start a fresh conversation\n"
        "❓ /help — Show this message\n\n"
        "Or just *chat naturally* — I'll understand! 😊\n"
        "Try: _\"I want 2 Chicken Biryani\"_ or _\"Book a table for 4 on Friday\"_"
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

    # ── Platform selection callback ──────────────────────────────────────────
    if query.data in PLATFORM_CALLBACKS:
        platform = PLATFORM_CALLBACKS[query.data]
        await save_session(chat_id, {
            "ordering_platform": platform,
            "history": [],
        })

        # When MCP is enabled, route Zomato/Swiggy into the LangGraph
        # MCP agent — the user stays inside Telegram and orders through
        # the live MCP server.
        if platform in ("Zomato", "Swiggy") and settings.MCP_ENABLED:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            opening = (
                "🟥 *Connecting you to Zomato* 🍽️"
                if platform == "Zomato"
                else "🟧 *Connecting you to Swiggy* 🍽️"
            )
            await query.message.reply_text(
                opening,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_keyboard(),
            )
            try:
                reply = await get_mcp_response(
                    user_message=(
                        "Hi! I just opened the chat — please greet me and ask "
                        "what cuisine I'm in the mood for, or for my delivery area / pin code."
                    ),
                    chat_id=chat_id,
                    user_name=user_name,
                    platform=platform,
                )
            except Exception as e:
                logger.error("MCP agent open failed: %s", e, exc_info=True)
                reply = None

            if reply:
                try:
                    await query.message.reply_text(
                        reply,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=main_keyboard(),
                    )
                except Exception:
                    await query.message.reply_text(reply, reply_markup=main_keyboard())
                return
            # else fall through to redirect-link fallback

        # Fallback / MCP disabled → external app redirect
        if platform == "Zomato":
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🟥 Open Zomato", url=settings.ZOMATO_URL)],
                [InlineKeyboardButton(
                    "↩️ Order via Dzukku Bot instead",
                    callback_data="platform_dzukku",
                )],
            ])
            await query.message.reply_text(
                "🟥 *Zomato selected!*\n\n"
                "Tap below to open our Zomato page and place your order on the app.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
            )
            return

        if platform == "Swiggy":
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🟧 Open Swiggy", url=settings.SWIGGY_URL)],
                [InlineKeyboardButton(
                    "↩️ Order via Dzukku Bot instead",
                    callback_data="platform_dzukku",
                )],
            ])
            await query.message.reply_text(
                "🟧 *Swiggy selected!*\n\n"
                "Tap below to open our Swiggy page and place your order on the app.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
            )
            return

        # Dzukku → continue inside the bot
        await query.message.reply_text(
            "🍽️ *Awesome — let's get you sorted right here!* ❤️",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_keyboard(),
        )
        await query.message.reply_text(
            "Quick actions 👇",
            reply_markup=quick_actions_inline(),
        )
        return

    intent = CALLBACK_INTENT_MAP.get(query.data)
    if not intent:
        return

    await _think_and_reply(update, context, intent)


# ── Text message handler ──────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text or ""

    # Greeting → show the platform-selection prompt only before a platform is
    # chosen. External MCP carts live on Zomato/Swiggy, not in the local Dzukku
    # cart, so using the local cart as the only "mid-flow" signal breaks MCP
    # conversations by re-opening the picker on a simple "hi".
    if _is_greeting(raw) and raw not in BUTTON_INTENT_MAP:
        chat_id = update.effective_chat.id
        sess    = await get_session(chat_id)
        cart    = sess.get("cart", []) or []
        platform_chosen = (sess.get("ordering_platform") or "").strip()

        # Only re-prompt before platform selection. Once Zomato/Swiggy is
        # selected, route greetings into that MCP agent so its session remains
        # continuous across the chat.
        if not cart and not platform_chosen:
            user      = update.effective_user
            user_name = (user.first_name if user else "") or sess.get("user_name") or "there"

            greeting_line = f"👋 Hey *{user_name}*! Welcome to *Dzukku Restaurant* 🍽️"
            await update.message.reply_text(
                greeting_line,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_keyboard(),
            )
            await _send_platform_prompt(update)
            return

    # Map keyboard buttons to natural language
    message = BUTTON_INTENT_MAP.get(raw, raw)
    await _think_and_reply(update, context, message)


# ── Location message handler ──────────────────────────────────────────────────

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    location  = update.message.location
    lat, lng  = location.latitude, location.longitude
    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name if user else ""

    location_message = (
        f"I'm at coordinates: lat {lat:.6f}, lng {lng:.6f}. "
        "Find restaurants near me."
    )

    await _think_and_reply(update, context, location_message)


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
    app.add_handler(CommandHandler("reserve", cmd_reserve))
    app.add_handler(CommandHandler("cart",    cmd_cart))
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("help",    cmd_help))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.add_error_handler(error_handler)
    return app
