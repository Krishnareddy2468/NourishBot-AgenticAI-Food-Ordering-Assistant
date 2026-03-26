import asyncio
import logging
import os
import re
from telegram import Update, Bot
from telegram.error import NetworkError, BadRequest
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
)
from app.services.langgraph_agent import langgraph_food_agent
from app.services.order_service import order_service
from app.services.session_service import session_service
from app.services.voice_service import voice_service
from app.config import app_config

logger = logging.getLogger(__name__)
QR_MARKER_RE = re.compile(r"\[QR Code Image Saved to ([^\]]+)\]")


class TelegramBotService:
    def __init__(self):
        self.token = app_config.telegram_bot_token
        self.app = None
        self.bot = None
        self._running = False
        self._last_error: str | None = None

        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN not configured - bot won't start")
            return

        self.bot = Bot(token=self.token)

    async def start(self):
        if not self.token:
            logger.warning("Skipping telegram bot - no valid token set")
            self._last_error = "TELEGRAM_BOT_TOKEN not configured"
            return

        try:
            self.app = Application.builder().token(self.token).build()

            # register all the handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("help", self._handle_help))
            self.app.add_handler(CommandHandler("menu", self._handle_menu))
            self.app.add_handler(CommandHandler("cart", self._handle_cart))
            self.app.add_handler(CommandHandler("status", self._handle_status))
            self.app.add_handler(CommandHandler("reset", self._handle_reset))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
            self.app.add_handler(MessageHandler(filters.LOCATION, self._handle_location))
            self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

            logger.info("Telegram bot starting polling...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)
            self._running = True
            self._last_error = None
            logger.info("Telegram bot is running")

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            self._running = False
            self._last_error = str(e)

    async def stop(self):
        if self.app and self._running:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                self._running = False
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
                self._last_error = str(e)

    async def send_message(self, user_id: str, text: str):
        """send a message to a user - used for order status notifications"""
        if not self.bot:
            return
        try:
            max_len = 4000
            if len(text) > max_len:
                # telegram has a character limit so we split long messages
                parts = [text[i:i + max_len] for i in range(0, len(text), max_len)]
                for part in parts:
                    await self.bot.send_message(chat_id=int(user_id), text=part, parse_mode="Markdown")
            else:
                await self.bot.send_message(chat_id=int(user_id), text=text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send message to {user_id}: {e}")
            # sometimes markdown parsing fails, retry as plain text
            try:
                await self.bot.send_message(chat_id=int(user_id), text=text)
            except Exception as e2:
                logger.error(f"Even plain text failed: {e2}")

    # -- command handlers --

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_id = str(user.id)
            user_name = user.first_name or user.username or ""
            response, _ = await langgraph_food_agent.process_message(user_id, "hi", user_name, channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            response, _ = await langgraph_food_agent.process_message(user_id, "help", channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            response, _ = await langgraph_food_agent.process_message(user_id, "show menu", channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_cart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            response, _ = await langgraph_food_agent.process_message(user_id, "show cart", channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            response, _ = await langgraph_food_agent.process_message(user_id, "order status", channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = str(update.effective_user.id)
            response, _ = await langgraph_food_agent.process_message(user_id, "start over", channel="telegram", input_mode="text")
            await self._send_response(update, response)
        finally:
            session_service.save()

    # -- message handlers --

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_id = str(user.id)
            user_name = user.first_name or user.username or ""
            message = update.message.text

            logger.info(f"[{user_name}] ({user_id}): {message}")
            await self._safe_chat_action(update, "typing")

            response, _ = await langgraph_food_agent.process_message(
                user_id=user_id,
                message=message,
                user_name=user_name,
                channel="telegram",
                input_mode="text",
            )
            logger.info(f"Response to {user_name}: {response[:80]}...")
            await self._send_response(update, response)

            session = session_service.get_session(user_id)
            if session.current_order_id:
                order = order_service.get_order(session.current_order_id)
                if order and order.status.value == "confirmed":
                    asyncio.create_task(
                        order_service.simulate_order_progress(
                            session.current_order_id,
                            send_update_callback=self._send_order_update,
                        )
                    )
        finally:
            session_service.save()

    async def _handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_id = str(user.id)
            loc = update.message.location

            address = f"lat={loc.latitude:.6f}, lon={loc.longitude:.6f}"
            response, _ = await langgraph_food_agent.process_message(
                user_id=user_id,
                message=f"My location is {address}",
                user_location=address,
                channel="telegram",
                input_mode="text",
            )
            await self._send_response(update, response)
        finally:
            session_service.save()

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_id = str(user.id)
            user_name = user.first_name or user.username or ""
            if not update.message or not update.message.voice:
                return

            await self._safe_chat_action(update, "typing")
            voice = update.message.voice
            mime_type = voice.mime_type or "audio/ogg"
            file = await context.bot.get_file(voice.file_id)
            audio_bytes = await file.download_as_bytearray()
            transcript = await voice_service.transcribe_audio(bytes(audio_bytes), mime_type=mime_type)
            logger.info("Voice transcript for user %s: %s", user_id, transcript[:120])
            response, _ = await langgraph_food_agent.process_message(
                user_id=user_id,
                message=transcript,
                user_name=user_name,
                channel="telegram",
                input_mode="voice",
            )
            await self._send_response(update, f"🎙️ {transcript}\n\n{response}")
        except Exception as e:
            logger.error("Voice handling failed for user %s: %s", user_id, e)
            await self._send_response(
                update,
                "I could not process that voice note. Please try a shorter voice note or send text.",
            )
        finally:
            session_service.save()

    # -- utils --

    async def _safe_chat_action(self, update: Update, action: str):
        if not update.message or not update.message.chat:
            return
        try:
            await update.message.chat.send_action(action)
        except NetworkError as e:
            logger.error("Telegram network error while sending chat action: %s", e)
        except Exception as e:
            logger.error("Telegram chat action failed: %s", e)

    async def _safe_reply_text(self, update: Update, text: str, parse_mode: str | None = None):
        if not update.message:
            return
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
        except NetworkError as e:
            logger.error("Telegram network error while sending reply: %s", e)
            self._last_error = str(e)
            if parse_mode is not None and "can't parse entities" in str(e).lower():
                try:
                    await update.message.reply_text(text)
                except Exception as inner:
                    logger.error("Telegram plain reply after entity parse failure failed: %s", inner)
                    self._last_error = str(inner)
        except BadRequest as e:
            logger.error("Telegram rejected formatted reply: %s", e)
            self._last_error = str(e)
            if parse_mode is not None:
                try:
                    await update.message.reply_text(text)
                except NetworkError as inner:
                    logger.error("Telegram network error while sending plain reply: %s", inner)
                    self._last_error = str(inner)
                except Exception as inner:
                    logger.error("Telegram plain reply failed: %s", inner)
                    self._last_error = str(inner)
        except Exception:
            if parse_mode is not None:
                try:
                    await update.message.reply_text(text)
                except NetworkError as e:
                    logger.error("Telegram network error while sending plain reply: %s", e)
                    self._last_error = str(e)
                except Exception as inner:
                    logger.error("Telegram reply failed: %s", inner)
                    self._last_error = str(inner)
            else:
                raise

    async def _send_response(self, update: Update, text: str):
        """try to send with markdown first, fall back to plain if it fails"""
        try:
            qr_match = QR_MARKER_RE.search(text or "")
            text_to_send = text
            if qr_match:
                qr_path = qr_match.group(1).strip()
                text_to_send = QR_MARKER_RE.sub("", text).strip()
                if os.path.exists(qr_path):
                    try:
                        with open(qr_path, "rb") as photo:
                            await update.message.reply_photo(photo=photo)
                    except Exception as e:
                        logger.error("Failed to send QR image %s: %s", qr_path, e)
            if not text_to_send:
                return

            max_len = 4000
            if len(text_to_send) > max_len:
                parts = [text_to_send[i:i + max_len] for i in range(0, len(text_to_send), max_len)]
                for part in parts:
                    await self._safe_reply_text(update, part, parse_mode="Markdown")
            else:
                await self._safe_reply_text(update, text_to_send, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            await self._safe_reply_text(update, "Sorry, something went wrong. Please try again!")

    async def _send_order_update(self, user_id: str, message: str):
        await self.send_message(user_id, message)


telegram_bot = TelegramBotService()
