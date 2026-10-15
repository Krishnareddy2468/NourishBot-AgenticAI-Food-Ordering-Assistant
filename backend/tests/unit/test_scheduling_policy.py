from datetime import datetime, timedelta

import pytz

from app.agent.policies import RestaurantPolicy


TZ = pytz.timezone("Asia/Kolkata")


def test_rejects_instant_order_inside_minimum_lead_time():
    policy = RestaurantPolicy(min_schedule_lead_minutes=30)
    now = TZ.localize(datetime(2026, 10, 15, 10, 0))

    valid, message = policy.validate_scheduled_for(now + timedelta(minutes=10), now)

    assert valid is False
    assert "30 minutes ahead" in message


def test_accepts_future_meal_during_operating_hours():
    policy = RestaurantPolicy(min_schedule_lead_minutes=30, max_schedule_days=14)
    now = TZ.localize(datetime(2026, 10, 15, 10, 0))

    valid, message = policy.validate_scheduled_for(now + timedelta(hours=3), now)

    assert valid is True
    assert message == ""


def test_rejects_schedule_outside_configured_window():
    policy = RestaurantPolicy(max_schedule_days=14)
    now = TZ.localize(datetime(2026, 10, 15, 10, 0))

    valid, message = policy.validate_scheduled_for(now + timedelta(days=15), now)

    assert valid is False
    assert "up to 14 days" in message


def test_allows_drive_through_fulfillment():
    policy = RestaurantPolicy(drive_thru_available=True)
    assert policy.validate_order_type("DRIVE_THRU") == (True, "")


def test_allows_home_delivery_fulfillment():
    policy = RestaurantPolicy(delivery_available=True)
    assert policy.validate_order_type("DELIVERY") == (True, "")
