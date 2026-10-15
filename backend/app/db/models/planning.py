"""Meal-planning domain records and immutable lifecycle histories."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text

from app.db.base import Base, RestaurantMixin


class MealPlan(Base, RestaurantMixin):
    __tablename__ = "meal_plans"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    status = Column(Text, nullable=False, server_default=text("'DRAFT'"))
    starts_on = Column(Date, nullable=False)
    ends_on = Column(Date, nullable=False)
    timezone = Column(Text, nullable=False, server_default=text("'Asia/Kolkata'"))
    created_by_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MealTemplate(Base, RestaurantMixin):
    __tablename__ = "meal_templates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    scope = Column(Text, nullable=False, server_default=text("'MEAL'"))
    definition_json = Column(JSONB, nullable=False, default=dict)
    active = Column(Boolean, nullable=False, server_default=text("true"))
    times_used = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MealRoutine(Base, RestaurantMixin):
    __tablename__ = "meal_routines"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    template_id = Column(BigInteger, ForeignKey("meal_templates.id", ondelete="RESTRICT"), nullable=False)
    name = Column(Text, nullable=False)
    state = Column(Text, nullable=False, server_default=text("'ACTIVE'"))
    meal_period = Column(Text, nullable=False)
    timezone = Column(Text, nullable=False, server_default=text("'Asia/Kolkata'"))
    recurrence_rule = Column(Text, nullable=False)
    next_materialization_at = Column(DateTime(timezone=True), index=True)
    paused_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MealBudget(Base, RestaurantMixin):
    __tablename__ = "meal_budgets"
    __table_args__ = (
        UniqueConstraint(
            "restaurant_id",
            "customer_id",
            "period",
            "starts_on",
            name="uq_meal_budget_customer_period_start",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    period = Column(Text, nullable=False)
    limit_cents = Column(Integer, nullable=False)
    currency = Column(Text, nullable=False, server_default=text("'INR'"))
    starts_on = Column(Date, nullable=False)
    ends_on = Column(Date, nullable=False)
    alert_thresholds_json = Column(JSONB, nullable=False, default=list)
    active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MealSchedule(Base, RestaurantMixin):
    __tablename__ = "meal_schedules"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    meal_plan_id = Column(BigInteger, ForeignKey("meal_plans.id", ondelete="SET NULL"), index=True)
    template_id = Column(BigInteger, ForeignKey("meal_templates.id", ondelete="SET NULL"))
    routine_id = Column(BigInteger, ForeignKey("meal_routines.id", ondelete="SET NULL"), index=True)
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), unique=True, index=True)
    title = Column(Text, nullable=False)
    meal_period = Column(Text, nullable=False, server_default=text("'UNSPECIFIED'"))
    state = Column(Text, nullable=False, server_default=text("'DRAFT'"), index=True)
    desired_for = Column(DateTime(timezone=True), nullable=False, index=True)
    ready_by = Column(DateTime(timezone=True))
    submission_window_start = Column(DateTime(timezone=True))
    submission_window_end = Column(DateTime(timezone=True))
    timezone = Column(Text, nullable=False, server_default=text("'Asia/Kolkata'"))
    source = Column(Text, nullable=False, server_default=text("'WEB'"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class OrderEvent(Base, RestaurantMixin):
    """Append-only audit record for an order or provider lifecycle change."""

    __tablename__ = "order_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(Text, nullable=False, index=True)
    from_state = Column(Text)
    to_state = Column(Text, nullable=False)
    actor_type = Column(Text, nullable=False)
    actor_id = Column(Text)
    source = Column(Text, nullable=False)
    reason_code = Column(Text)
    metadata_json = Column(JSONB, nullable=False, default=dict)
    occurred_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)


class ScheduleEvent(Base, RestaurantMixin):
    """Append-only audit record for a meal schedule lifecycle change."""

    __tablename__ = "schedule_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    meal_schedule_id = Column(
        BigInteger,
        ForeignKey("meal_schedules.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), index=True)
    event_type = Column(Text, nullable=False, index=True)
    from_state = Column(Text)
    to_state = Column(Text, nullable=False)
    actor_type = Column(Text, nullable=False)
    actor_id = Column(Text)
    source = Column(Text, nullable=False)
    reason_code = Column(Text)
    metadata_json = Column(JSONB, nullable=False, default=dict)
    occurred_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)


def _reject_event_mutation(_mapper, _connection, target) -> None:
    raise ValueError(f"{target.__class__.__name__} records are append-only")


for _event_model in (OrderEvent, ScheduleEvent):
    event.listen(_event_model, "before_update", _reject_event_mutation)
    event.listen(_event_model, "before_delete", _reject_event_mutation)
