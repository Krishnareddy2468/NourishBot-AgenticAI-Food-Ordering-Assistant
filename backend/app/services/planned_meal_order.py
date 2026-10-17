"""Planned-meal order handoff shared by chat commands and the customer web dashboard.

Both entry points must apply the same guards before a confirmed planned meal may
become an individual order: plan and meal confirmation, a future meal time, live
availability, unchanged dietary metadata, and unchanged verified nutrition facts.
The web dashboard then reuses the chat confirmation challenge, idempotency key,
and order creation path instead of carrying its own checkout rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select

from app.db.models import Customer, MealPlan, MealSchedule, MenuItem, Order
from app.db.models import Session as SessionModel
from app.db.session import AsyncSessionLocal
from app.domain.meal_planning import NUTRITION_TARGET_KEYS, MenuCandidate

# Web checkout covers the in-house kitchen only. An external provider order stays
# behind the provider capability gate and its chat conversation.
IN_HOUSE_PROVIDER = "NOURISHBOT"
WEB_ORDER_TYPES = ("DELIVERY", "PICKUP")
PLACEHOLDER_PHONE_PREFIX = "web-account-pending-"


class PlannedMealOrderError(RuntimeError):
    """A planned meal cannot be handed to the order flow as it currently stands."""


@dataclass(frozen=True)
class OrderablePlannedMeal:
    """A confirmed planned meal that still matches the live menu."""

    meal: Any
    plan: Any
    menu_item: Any

    @property
    def scheduled_local(self) -> datetime:
        return self.meal.desired_for.astimezone(ZoneInfo(self.plan.timezone))

    @property
    def scheduled_text(self) -> str:
        return f"{self.scheduled_local.isoformat()} ({self.plan.timezone})"

    @property
    def planning_context(self) -> str:
        """The exact-order instructions handed to the chat ordering flow."""
        return (
            f"Meal #{self.meal.id} from confirmed plan #{self.plan.id}; selected menu item "
            f"{self.menu_item.name!r} (internal menu item ID {self.menu_item.id}, not an external provider ID); "
            "quantity 1; fulfillment DELIVERY; "
            f"customer-requested future time {self.scheduled_text}; "
            f"plan estimate {self.meal.currency} {self.meal.estimated_cost_cents / 100:.2f}. "
            "Recheck live availability and current price. If current price differs from the "
            "plan estimate, show the current price in the exact order confirmation."
        )


async def load_orderable_planned_meal(
    db,
    *,
    meal_id: int,
    restaurant_id: int,
    customer_id: int,
) -> OrderablePlannedMeal:
    """Return one planned meal that may be ordered, or explain why it may not.

    Every refusal is deliberate: the customer reviews and reconfirms the plan
    rather than the system silently ordering against changed data.
    """
    meal = (await db.execute(select(MealSchedule).where(
        MealSchedule.id == meal_id,
        MealSchedule.restaurant_id == restaurant_id,
        MealSchedule.customer_id == customer_id,
    ))).scalar_one_or_none()
    if not meal:
        raise PlannedMealOrderError("That planned meal was not found on your linked profile.")
    plan = (await db.execute(select(MealPlan).where(
        MealPlan.id == meal.meal_plan_id,
        MealPlan.restaurant_id == restaurant_id,
        MealPlan.customer_id == customer_id,
    ))).scalar_one_or_none()
    if not plan:
        raise PlannedMealOrderError("That meal plan was not found on your linked profile.")
    if meal.order_id:
        raise PlannedMealOrderError(
            "That planned meal was already ordered. Check your orders, or plan another meal."
        )
    if meal.state != "CONFIRMED" or plan.status != "CONFIRMED":
        raise PlannedMealOrderError(
            f"Confirm and review plan #{plan.id} first. Plan confirmation only confirms "
            "the schedule; it does not place an order."
        )
    if meal.desired_for <= datetime.now(ZoneInfo("UTC")):
        raise PlannedMealOrderError(
            "That planned meal time has passed. Reschedule and reconfirm it before ordering."
        )
    if not meal.menu_item_id:
        raise PlannedMealOrderError(
            "That planned meal has no linked menu item. Edit the plan before ordering."
        )

    menu_item = (await db.execute(select(MenuItem).where(
        MenuItem.id == meal.menu_item_id,
        MenuItem.restaurant_id == restaurant_id,
        MenuItem.available.is_(True),
        or_(MenuItem.stock_qty.is_(None), MenuItem.stock_qty > 0),
    ))).scalar_one_or_none()
    if not menu_item:
        raise PlannedMealOrderError(
            "That menu item is no longer available. Edit and reconfirm the plan before ordering."
        )

    selection = meal.selection_json or {}
    saved_type = str(selection.get("menu_type") or "").upper()
    current_candidate = MenuCandidate.from_mapping({
        "id": menu_item.id,
        "name": menu_item.name,
        "price_cents": menu_item.price_cents,
        "menu_type": menu_item.type or "",
        "tags": menu_item.tags or [],
    })
    current_type = current_candidate.menu_type
    saved_tags = set(meal.diet_tags or [])
    current_tags = {current_type.lower(), *current_candidate.tags}
    saved_allergens = set(meal.allergen_tags or [])
    current_allergens = {
        tag for tag in current_candidate.tags
        if tag.startswith(("allergen-", "contains-"))
    }
    if saved_type != current_type or saved_tags != current_tags or saved_allergens != current_allergens:
        raise PlannedMealOrderError(
            "The menu or dietary metadata changed. Review and reconfirm this meal before ordering."
        )

    constraints = plan.constraints_json or {}
    if constraints.get("nutrition_targets"):
        saved_nutrition = selection.get("nutrition") or {}
        current_values = {
            "energy_kcal": menu_item.energy_kcal,
            "protein_g": menu_item.protein_g,
            "carbohydrates_g": menu_item.carbohydrates_g,
            "fat_g": menu_item.fat_g,
            "fiber_g": menu_item.fiber_g,
        }
        saved_values = saved_nutrition.get("per_serving") or {}
        if (
            menu_item.nutrition_status != "VERIFIED"
            or menu_item.nutrition_source != saved_nutrition.get("source")
            or any(
                current_values[key] is None
                or float(current_values[key]) != float(saved_values.get(key, -1))
                for key in NUTRITION_TARGET_KEYS
            )
            or float(menu_item.nutrition_serving_g or 0)
            != float(saved_nutrition.get("serving_g") or -1)
        ):
            raise PlannedMealOrderError(
                "Verified nutrition facts changed. Rebuild and reconfirm the plan before ordering."
            )

    return OrderablePlannedMeal(meal=meal, plan=plan, menu_item=menu_item)


# ── Customer web dashboard checkout ───────────────────────────────────────────

@dataclass(frozen=True)
class _WebCheckout:
    """Everything the executor tools need for one planned meal, already verified."""

    ctx: Any
    meal_id: int
    plan_id: int
    menu_item_id: int
    menu_item_name: str
    price_cents: int
    currency: str
    estimated_cost_cents: int
    scheduled_for: str
    scheduled_text: str
    timezone: str
    order_type: str
    delivery_address: str

    @property
    def tool_args(self) -> dict[str, Any]:
        return {
            "order_type": self.order_type,
            "address": self.delivery_address,
            "scheduled_for": self.scheduled_for,
            "provider": IN_HOUSE_PROVIDER,
        }

    @property
    def meal_summary(self) -> dict[str, Any]:
        return {
            "meal_id": self.meal_id,
            "plan_id": self.plan_id,
            "menu_item_id": self.menu_item_id,
            "menu_item_name": self.menu_item_name,
            "current_price_cents": self.price_cents,
            "plan_estimate_cents": self.estimated_cost_cents,
            "price_changed_since_plan": self.price_cents != self.estimated_cost_cents,
            "currency": self.currency,
            "scheduled_for": self.scheduled_for,
            "scheduled_local_text": self.scheduled_text,
            "timezone": self.timezone,
        }


def _resolve_contact_phone(customer: Customer, contact_phone: str) -> str | None:
    """Return the contact number to persist, or None when the profile already has one.

    A profile number is never silently replaced from the browser: re-identifying an
    account belongs to the verified chat-linking flow.
    """
    stored = str(customer.phone or "")
    supplied = str(contact_phone or "").strip()
    if stored and not stored.startswith(PLACEHOLDER_PHONE_PREFIX):
        if supplied and supplied != stored:
            raise PlannedMealOrderError(
                "Your profile already has a contact number. Change it from a linked "
                "WhatsApp or Telegram chat before ordering here."
            )
        return None
    if not supplied:
        raise PlannedMealOrderError(
            "Add the mobile number the kitchen and rider should use before ordering."
        )
    if not (supplied.isdigit() and 8 <= len(supplied) <= 15):
        raise PlannedMealOrderError("Enter a valid contact mobile number of 8 to 15 digits.")
    return supplied


async def _ensure_web_session(channel) -> None:
    """A confirmation challenge is bound to a persisted session for this channel."""
    async with AsyncSessionLocal() as db:
        existing = (await db.execute(select(SessionModel).where(
            SessionModel.channel_id == channel.id,
            SessionModel.restaurant_id == channel.restaurant_id,
        ))).scalars().first()
        if existing:
            return
        db.add(SessionModel(
            restaurant_id=channel.restaurant_id,
            channel_id=channel.id,
            state="IDLE",
            history_json={"meta": {}, "turns": []},
        ))
        await db.commit()


async def _prepare_web_checkout(
    *,
    channel,
    meal_id: int,
    order_type: str,
    delivery_address: str,
    contact_phone: str,
) -> _WebCheckout:
    from app.agent.context_builder import build_context

    restaurant_id = int(channel.restaurant_id)
    customer_id = int(channel.customer_id)
    order_type = str(order_type or "").strip().upper()
    delivery_address = str(delivery_address or "").strip()
    if order_type not in WEB_ORDER_TYPES:
        raise PlannedMealOrderError("Choose home delivery or pickup for this planned meal.")
    if order_type == "DELIVERY" and not delivery_address:
        raise PlannedMealOrderError("A delivery address is required for a home delivery order.")

    async with AsyncSessionLocal() as db:
        planned = await load_orderable_planned_meal(
            db,
            meal_id=meal_id,
            restaurant_id=restaurant_id,
            customer_id=customer_id,
        )
        customer = (await db.execute(select(Customer).where(
            Customer.id == customer_id,
            Customer.restaurant_id == restaurant_id,
        ).with_for_update())).scalar_one_or_none()
        if not customer:
            raise PlannedMealOrderError("Your customer profile could not be loaded.")
        new_phone = _resolve_contact_phone(customer, contact_phone)
        if new_phone:
            duplicate = (await db.execute(select(Customer.id).where(
                Customer.restaurant_id == restaurant_id,
                Customer.phone == new_phone,
                Customer.id != customer.id,
            ))).scalar_one_or_none()
            if duplicate:
                raise PlannedMealOrderError(
                    "That mobile number already belongs to another NourishBot profile. "
                    "Link that profile from its chat, or use a different number."
                )
            customer.phone = new_phone
        customer_name = str(customer.name or "").strip()
        if not customer_name:
            raise PlannedMealOrderError("Add your name to your profile before ordering.")
        details = {
            "customer_name": customer_name,
            "meal_id": int(planned.meal.id),
            "plan_id": int(planned.plan.id),
            "menu_item_id": int(planned.menu_item.id),
            "menu_item_name": str(planned.menu_item.name),
            "price_cents": int(planned.menu_item.price_cents),
            "menu_item_type": str(planned.menu_item.type or ""),
            "currency": str(planned.meal.currency or "INR"),
            "estimated_cost_cents": int(planned.meal.estimated_cost_cents or 0),
            "scheduled_for": planned.meal.desired_for.astimezone(timezone.utc).isoformat(),
            "scheduled_text": planned.scheduled_text,
            "timezone": str(planned.plan.timezone),
        }
        await db.commit()

    await _ensure_web_session(channel)
    ctx = await build_context(
        chat_id=channel.external_id,
        user_name=details["customer_name"],
        channel_type="WEB",
    )
    if not ctx.session_id:
        raise PlannedMealOrderError("Your web session is not ready. Reload the dashboard and try again.")
    if ctx.customer_id != customer_id:
        raise PlannedMealOrderError("Your web session is linked to a different profile. Sign in again.")

    provider = str(ctx.ordering_platform or IN_HOUSE_PROVIDER).upper()
    if provider != IN_HOUSE_PROVIDER:
        from app.providers.service import provider_experience

        experience = await provider_experience(provider)
        raise PlannedMealOrderError(
            f"{provider.title()} is in {experience['mode_label']}, and dashboard checkout places "
            f"{IN_HOUSE_PROVIDER.title()} kitchen orders only. Order this meal from your linked "
            "WhatsApp or Telegram chat instead. No order was placed."
        )
    if ctx.cart:
        raise PlannedMealOrderError(
            "An in-progress cart is already open on your account. Finish or clear it first so this "
            "planned meal cannot accidentally be combined with other items. No order was placed."
        )
    if order_type == "DELIVERY" and details["price_cents"] < ctx.policy.min_order_for_delivery_cents:
        minimum = ctx.policy.min_order_for_delivery_cents // 100
        raise PlannedMealOrderError(
            f"Minimum order for delivery is ₹{minimum}. This meal is "
            f"₹{details['price_cents'] // 100}. Choose pickup or order it with other items in chat."
        )

    ctx.cart = [{
        "item_id": details["menu_item_id"],
        "item_name": details["menu_item_name"],
        "qty": 1,
        "unit_price_cents": details["price_cents"],
        "available": True,
        "type": details["menu_item_type"],
    }]
    ctx.cart_total_cents = details["price_cents"]
    ctx.order_type = order_type
    ctx.delivery_address = delivery_address
    ctx.scheduled_for = details["scheduled_for"]
    ctx.ordering_platform = IN_HOUSE_PROVIDER

    return _WebCheckout(
        ctx=ctx,
        meal_id=details["meal_id"],
        plan_id=details["plan_id"],
        menu_item_id=details["menu_item_id"],
        menu_item_name=details["menu_item_name"],
        price_cents=details["price_cents"],
        currency=details["currency"],
        estimated_cost_cents=details["estimated_cost_cents"],
        scheduled_for=details["scheduled_for"],
        scheduled_text=details["scheduled_text"],
        timezone=details["timezone"],
        order_type=order_type,
        delivery_address=delivery_address,
    )


async def preview_planned_meal_web_order(
    *,
    channel,
    meal_id: int,
    order_type: str,
    delivery_address: str = "",
    contact_phone: str = "",
) -> dict[str, Any]:
    """Build the exact order summary a later explicit confirmation may place."""
    from app.agent.executor import ExecutionResult, _tool_prepare_order_confirmation

    checkout = await _prepare_web_checkout(
        channel=channel,
        meal_id=meal_id,
        order_type=order_type,
        delivery_address=delivery_address,
        contact_phone=contact_phone,
    )
    action = await _tool_prepare_order_confirmation(
        checkout.tool_args,
        checkout.ctx,
        ExecutionResult(),
    )
    if not action.ok:
        raise PlannedMealOrderError(action.error or "This planned meal cannot be ordered right now.")
    return {
        **action.data,
        **checkout.meal_summary,
        "message": (
            "Review this exact order and confirm it to place it. Nothing has been ordered yet."
        ),
    }


async def _order_for_confirmation(
    *,
    restaurant_id: int,
    customer_id: int,
    confirmation_id: str,
) -> dict[str, Any] | None:
    """Return the order this confirmation already placed, if it placed one.

    A resubmitted confirmation — a double click, or a retried request — must return
    the first order rather than a refusal or a second order.
    """
    if not confirmation_id:
        return None
    async with AsyncSessionLocal() as db:
        order = (await db.execute(select(Order).where(
            Order.restaurant_id == restaurant_id,
            Order.customer_id == customer_id,
            Order.confirmation_id == confirmation_id,
        ))).scalars().first()
        if not order:
            return None
        return {
            "order_ref": order.order_ref,
            "order_type": order.order_type,
            "scheduled_for": order.scheduled_for.isoformat() if order.scheduled_for else None,
            "provider": order.provider,
            "tracking_token": order.tracking_token,
            "idempotent_replay": True,
            "message": "This order was already placed from your meal plan.",
        }


async def place_planned_meal_web_order(
    *,
    channel,
    meal_id: int,
    confirmation_id: str,
    order_type: str,
    delivery_address: str = "",
) -> dict[str, Any]:
    """Place the reviewed planned-meal order against an unused confirmation."""
    from app.agent.executor import ExecutionResult, _tool_place_order

    confirmation_id = str(confirmation_id or "").strip()
    replay = await _order_for_confirmation(
        restaurant_id=int(channel.restaurant_id),
        customer_id=int(channel.customer_id),
        confirmation_id=confirmation_id,
    )
    if replay:
        return {**replay, "meal_id": int(meal_id)}

    checkout = await _prepare_web_checkout(
        channel=channel,
        meal_id=meal_id,
        order_type=order_type,
        delivery_address=delivery_address,
        contact_phone="",
    )
    action = await _tool_place_order(
        {
            **checkout.tool_args,
            "confirmation_id": confirmation_id,
            "meal_schedule_id": checkout.meal_id,
            "notes": f"Planned meal #{checkout.meal_id} from meal plan #{checkout.plan_id}",
        },
        checkout.ctx,
        ExecutionResult(),
    )
    if not action.ok:
        raise PlannedMealOrderError(action.error or "This planned meal could not be ordered.")
    return {
        **action.data,
        **checkout.meal_summary,
        "message": "Order confirmed from your meal plan.",
    }
