"""Periodic bounded materialization of active meal routines."""

import asyncio
import logging

from app.services.meal_planner import materialize_due_routines
from app.workers.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.meal_routine_worker.materialize_meal_routines")
def materialize_meal_routines() -> dict[str, int]:
    """Materialize each active routine only through its controlled horizon."""
    try:
        return asyncio.run(materialize_due_routines())
    except Exception:
        logger.exception("Meal routine materialization failed")
        raise
