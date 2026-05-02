"""
Google Sheets sync — best-effort, never blocks the main flow.
Callers should wrap in try/except or use the safe_* helpers below.
"""

import json
import logging
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

from app.core.config import settings

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_sheet():
    if settings.GOOGLE_CREDENTIALS:
        creds_dict = json.loads(settings.GOOGLE_CREDENTIALS)
        creds = Credentials.from_service_account_info(creds_dict, scopes=_SCOPES)
    else:
        creds = Credentials.from_service_account_file(
            str(settings.CREDS_PATH), scopes=_SCOPES
        )
    client = gspread.authorize(creds)
    return client.open_by_key(settings.GOOGLE_SHEET_ID)


def sync_order(
    customer_name:  str,
    customer_phone: str,
    items_summary:  str,
    total_price:    float,
    order_ref:      str = "",
) -> None:
    sheet  = _get_sheet()
    ws     = sheet.worksheet("Orders")
    next_id = len(ws.get_all_values())
    ws.append_row([
        order_ref or next_id,
        customer_name,
        customer_phone,
        items_summary,
        f"₹{total_price}",
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Pending",
    ])
    logger.info("Order synced to Sheets: %s", order_ref)


def sync_reservation(
    customer_name:   str,
    customer_phone:  str,
    date:            str,
    time:            str,
    guests:          int,
    special_request: str = "",
    reservation_ref: str = "",
) -> None:
    sheet  = _get_sheet()
    ws     = sheet.worksheet("Reservations")
    next_id = len(ws.get_all_values())
    ws.append_row([
        reservation_ref or next_id,
        customer_name,
        customer_phone,
        date,
        time,
        guests,
        special_request,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Confirmed",
    ])
    logger.info("Reservation synced to Sheets: %s", reservation_ref)


# ── Safe wrappers (swallow exceptions, log warning) ───────────────────────────

def safe_sync_order(*args, **kwargs) -> None:
    try:
        sync_order(*args, **kwargs)
    except Exception as e:
        logger.warning("Sheets order sync failed: %s", e)


def safe_sync_reservation(*args, **kwargs) -> None:
    try:
        sync_reservation(*args, **kwargs)
    except Exception as e:
        logger.warning("Sheets reservation sync failed: %s", e)


# Backward-compat aliases used by older callers
sync_order_to_sheet       = safe_sync_order
sync_reservation_to_sheet = safe_sync_reservation
