from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.memory_agent import (
    _apply_fulfillment_signal,
    _apply_rating_signal,
    can_learn_from_order,
)
from app.agent.privacy_service import (
    _masked_external_id,
    _create_link_code,
    _redeem_link_code,
    _token_hash,
    is_account_command,
    prune_session_history,
    prune_session_location_data,
)
from app.db.models import Channel, ConfirmationChallenge, UserPreferences
from app.api.routes.customer_privacy import LinkCodeRequest, _browser_identity


class _Result:
    def __init__(self, value=None):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _RecordingDb:
    def __init__(self, result=None):
        self.result = result
        self.added = []

    async def execute(self, _statement):
        return _Result(self.result)

    def add(self, value):
        self.added.append(value)


def test_privacy_commands_are_explicit_and_do_not_capture_normal_chat():
    assert is_account_command("enable memory") is True
    assert is_account_command("/privacy") is True
    assert is_account_command("link whatsapp") is True
    assert is_account_command("Can you remember less spicy food?") is False
    assert is_account_command("Please link me a restaurant") is False


def test_link_tokens_are_hashed_and_channel_ids_are_masked():
    raw = "one-time-secret"
    assert _token_hash(raw) != raw
    assert _token_hash(raw) == _token_hash(raw)
    assert _masked_external_id("919876543210") == "***3210"


def test_web_identity_is_opaque_and_link_code_input_is_bounded():
    raw = "browser-cookie-secret"
    assert _browser_identity(raw) != raw
    assert _browser_identity(raw) == _browser_identity(raw)
    with pytest.raises(ValidationError):
        LinkCodeRequest(code="short")


@pytest.mark.asyncio
async def test_link_challenge_is_hashed_expiring_and_target_channel_bound():
    db = _RecordingDb()
    source = Channel(
        id=3,
        restaurant_id=1,
        customer_id=7,
        type="TELEGRAM",
        external_id="123",
        verified_at=datetime.now(timezone.utc),
    )

    token = await _create_link_code(db, source, "WHATSAPP")
    challenge = db.added[0]

    assert challenge.token_hash == _token_hash(token)
    assert challenge.token_hash != token
    assert challenge.target_channel_type == "WHATSAPP"
    assert challenge.expires_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_redeeming_link_records_target_and_refuses_silent_account_merge():
    challenge = SimpleNamespace(
        customer_id=7,
        used_at=None,
        target_channel_id=None,
    )
    target = Channel(
        id=4,
        restaurant_id=1,
        customer_id=None,
        type="WHATSAPP",
        external_id="919999999999",
        verified_at=datetime.now(timezone.utc),
    )
    reply = await _redeem_link_code(_RecordingDb(challenge), target, "secret")

    assert reply.startswith("Your channel is now securely linked")
    assert target.customer_id == 7
    assert challenge.used_at is not None
    assert challenge.target_channel_id == 4

    conflicting = Channel(
        id=5,
        restaurant_id=1,
        customer_id=99,
        type="WHATSAPP",
        external_id="918888888888",
        verified_at=datetime.now(timezone.utc),
    )
    conflict_reply = await _redeem_link_code(
        _RecordingDb(SimpleNamespace(customer_id=7, used_at=None, target_channel_id=None)),
        conflicting,
        "secret",
    )
    assert "already linked to another customer" in conflict_reply


def test_channel_identity_is_unique_inside_tenant_and_channel_type():
    constraint = next(
        item
        for item in Channel.__table__.constraints
        if getattr(item, "name", None) == "uq_channel_restaurant_type_external"
    )
    assert [column.name for column in constraint.columns] == [
        "restaurant_id",
        "type",
        "external_id",
    ]


def test_confirmation_model_contains_phase1_audit_attribution():
    columns = ConfirmationChallenge.__table__.columns
    assert "created_by_channel_id" in columns
    assert "created_by_channel_type" in columns
    assert "confirmed_by_channel_id" in columns
    assert "invalidated_by_channel_id" in columns
    assert "confirmation_actor" in columns
    assert "invalidation_reason" in columns


def test_learning_accepts_only_fulfillment_or_explicit_valid_rating():
    assert can_learn_from_order("DELIVERED", "FULFILLMENT") is True
    assert can_learn_from_order("COMPLETED", "FULFILLMENT") is True
    assert can_learn_from_order("SCHEDULED", "FULFILLMENT") is False
    assert can_learn_from_order("CANCELLED", "FULFILLMENT") is False
    assert can_learn_from_order("CANCELLED", "EXPLICIT_FEEDBACK", 5) is True
    assert can_learn_from_order("DELIVERED", "EXPLICIT_FEEDBACK", 0) is False
    assert can_learn_from_order("DELIVERED", "IMPLICIT_CLICK") is False


def test_explicit_rating_does_not_count_as_a_fulfilled_order():
    prefs = UserPreferences(
        cuisine_weights={},
        order_timing={},
        craving_cycles={},
        total_orders=0,
        total_spent_cents=0,
    )
    items = [{"item_name": "Paneer curry", "qty": 2, "price_cents": 25000}]

    _apply_rating_signal(prefs, items, 5)

    assert prefs.cuisine_weights["curry"] == 0.2
    assert prefs.total_orders == 0
    assert prefs.total_spent_cents == 0


def test_fulfillment_updates_taste_and_completed_order_totals():
    prefs = UserPreferences(
        cuisine_weights={},
        order_timing={},
        craving_cycles={},
        total_orders=0,
        total_spent_cents=0,
    )
    items = [{"item_name": "Paneer curry", "qty": 2, "price_cents": 25000}]
    occurred_at = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)

    _apply_fulfillment_signal(prefs, items, occurred_at=occurred_at)

    assert prefs.total_orders == 1
    assert prefs.total_spent_cents == 50000
    assert prefs.cuisine_weights["curry"] > 0
    assert prefs.last_order_at == occurred_at


def test_message_retention_prunes_only_expired_turns_and_keeps_metadata():
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    cutoff = now - timedelta(days=30)
    history = {
        "meta": {"scheduled_for": "2026-10-20T13:00:00+05:30"},
        "turns": [
            {"role": "user", "content": "old", "at": "2026-08-01T00:00:00+00:00"},
            {"role": "assistant", "content": "recent", "at": "2026-10-14T00:00:00+00:00"},
        ],
    }

    retained = prune_session_history(history, cutoff, now)

    assert retained["meta"] == history["meta"]
    assert [turn["content"] for turn in retained["turns"]] == ["recent"]


def test_legacy_message_history_expires_using_session_timestamp():
    cutoff = datetime(2026, 9, 15, tzinfo=timezone.utc)
    old_session = datetime(2026, 8, 1, tzinfo=timezone.utc)
    recent_session = datetime(2026, 10, 1, tzinfo=timezone.utc)
    legacy = [{"role": "user", "content": "untimestamped"}]

    assert prune_session_history(legacy, cutoff, old_session) == []
    assert prune_session_history(legacy, cutoff, recent_session) == legacy


def test_location_retention_removes_expired_coordinates_and_transient_address():
    cutoff = datetime(2026, 10, 8, tzinfo=timezone.utc)
    history = {
        "meta": {"delivery_address": "12 Lake View Road", "order_type": "DELIVERY"},
        "turns": [
            {
                "role": "user",
                "content": "My current location is 17.123,78.123. Use it as my route origin.",
                "at": "2026-10-01T00:00:00+00:00",
            },
            {
                "role": "assistant",
                "content": "Where should I schedule your meal?",
                "at": "2026-10-01T00:00:01+00:00",
            },
        ],
    }

    retained = prune_session_location_data(
        history,
        cutoff,
        datetime(2026, 10, 1, tzinfo=timezone.utc),
    )

    assert "delivery_address" not in retained["meta"]
    assert retained["meta"]["order_type"] == "DELIVERY"
    assert [turn["role"] for turn in retained["turns"]] == ["assistant"]
