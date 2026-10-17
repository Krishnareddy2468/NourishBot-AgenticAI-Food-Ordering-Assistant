"""
Menu routes — CRUD for menu items, categories, images, modifiers.
"""

from datetime import datetime, timezone
import math

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models import MenuCategory, MenuItem, MenuItemImage
from app.auth.deps import require_manager, require_staff, tenant_id
from app.core.storage import upload_image

router = APIRouter(prefix="/api/v1/menu", tags=["menu"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class MenuNutritionFields(BaseModel):
    energy_kcal: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    protein_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    carbohydrates_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fat_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fiber_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    nutrition_serving_g: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    nutrition_source: str | None = Field(default=None, max_length=500)
    nutrition_status: str = "UNKNOWN"


class MenuItemCreate(MenuNutritionFields):
    name: str
    description: Optional[str] = None
    type: Optional[str] = None  # VEG | NON_VEG | EGG | VEGAN
    price_cents: int
    special_price_cents: Optional[int] = None
    category_id: Optional[int] = None
    available: bool = True
    stock_qty: Optional[int] = None
    prep_time_sec: int = 900
    tags: Optional[list[str]] = None


class MenuItemUpdate(MenuNutritionFields):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    price_cents: Optional[int] = None
    special_price_cents: Optional[int] = None
    category_id: Optional[int] = None
    available: Optional[bool] = None
    stock_qty: Optional[int] = None
    prep_time_sec: Optional[int] = None
    tags: Optional[list[str]] = None


def _nutrition_data(data: dict, existing: MenuItem | None = None) -> dict:
    nutrition_keys = (
        "energy_kcal", "protein_g", "carbohydrates_g", "fat_g", "fiber_g",
        "nutrition_serving_g", "nutrition_source",
    )
    current = {
        key: getattr(existing, key, None)
        for key in nutrition_keys
    } if existing else {key: None for key in nutrition_keys}
    merged = {**current, **{key: value for key, value in data.items() if key in nutrition_keys}}
    status_value = data.get(
        "nutrition_status",
        existing.nutrition_status if existing else "UNKNOWN",
    )
    nutrition_status = str(status_value or "UNKNOWN").upper()
    if nutrition_status not in {"UNKNOWN", "ESTIMATED", "VERIFIED"}:
        raise HTTPException(422, "Nutrition status must be UNKNOWN, ESTIMATED, or VERIFIED")

    values = [merged[key] for key in ("energy_kcal", "protein_g", "carbohydrates_g", "fat_g", "fiber_g")]
    has_any_values = any(value is not None for value in values)
    if nutrition_status == "UNKNOWN":
        if has_any_values or merged["nutrition_serving_g"] is not None or merged["nutrition_source"]:
            raise HTTPException(422, "UNKNOWN nutrition status cannot include nutrition values or a source")
        merged = {key: None for key in nutrition_keys}
    elif nutrition_status == "ESTIMATED":
        if not has_any_values or not str(merged["nutrition_source"] or "").strip():
            raise HTTPException(422, "ESTIMATED nutrition requires at least one value and a source")
    elif any(value is None for value in values) or merged["nutrition_serving_g"] is None or not str(merged["nutrition_source"] or "").strip():
        raise HTTPException(
            422,
            "VERIFIED nutrition requires all five values, a serving size, and a source",
        )

    for key in ("energy_kcal", "protein_g", "carbohydrates_g", "fat_g", "fiber_g", "nutrition_serving_g"):
        value = merged[key]
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise HTTPException(422, f"{key} must be a finite non-negative number")
    if merged["nutrition_serving_g"] is not None and float(merged["nutrition_serving_g"]) == 0:
        raise HTTPException(422, "Nutrition serving size must be greater than zero")
    source = str(merged["nutrition_source"] or "").strip()
    merged["nutrition_source"] = source or None
    merged["nutrition_status"] = nutrition_status
    merged["nutrition_updated_at"] = datetime.now(timezone.utc) if any(
        key in data for key in (*nutrition_keys, "nutrition_status")
    ) or existing is None else existing.nutrition_updated_at
    return merged


class AvailabilityUpdate(BaseModel):
    available: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/items")
async def list_menu_items(user=Depends(require_staff)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MenuItem).where(MenuItem.restaurant_id == rid).order_by(MenuItem.name)
        )
        items = result.scalars().all()
        return {
            "items": [
                {
                    "id": i.id,
                    "name": i.name,
                    "description": i.description,
                    "type": i.type,
                    "price_cents": i.price_cents,
                    "special_price_cents": i.special_price_cents,
                    "available": i.available,
                    "stock_qty": i.stock_qty,
                    "category_id": i.category_id,
                    "tags": i.tags,
                    "energy_kcal": i.energy_kcal,
                    "protein_g": i.protein_g,
                    "carbohydrates_g": i.carbohydrates_g,
                    "fat_g": i.fat_g,
                    "fiber_g": i.fiber_g,
                    "nutrition_serving_g": i.nutrition_serving_g,
                    "nutrition_source": i.nutrition_source,
                    "nutrition_status": i.nutrition_status,
                    "nutrition_updated_at": i.nutrition_updated_at,
                }
                for i in items
            ]
        }


@router.post("/items", status_code=201)
async def create_menu_item(body: MenuItemCreate, user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        nutrition = _nutrition_data(body.model_dump())
        if body.category_id is not None:
            category = (await session.execute(
                select(MenuCategory).where(
                    MenuCategory.id == body.category_id,
                    MenuCategory.restaurant_id == rid,
                )
            )).scalar_one_or_none()
            if not category:
                raise HTTPException(404, "Menu category not found")
        item = MenuItem(
            restaurant_id=rid,
            name=body.name,
            description=body.description,
            type=body.type,
            price_cents=body.price_cents,
            special_price_cents=body.special_price_cents,
            category_id=body.category_id,
            available=body.available,
            stock_qty=body.stock_qty,
            prep_time_sec=body.prep_time_sec,
            tags=body.tags,
            **nutrition,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return {"id": item.id, "name": item.name}


@router.patch("/items/{item_id}")
async def update_menu_item(item_id: int, body: MenuItemUpdate, user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(MenuItem).where(MenuItem.id == item_id, MenuItem.restaurant_id == rid))
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Menu item not found")

        update_data = body.model_dump(exclude_unset=True)
        nutrition_keys = {
            "energy_kcal", "protein_g", "carbohydrates_g", "fat_g", "fiber_g",
            "nutrition_serving_g", "nutrition_source", "nutrition_status",
        }
        nutrition_changed = bool(nutrition_keys.intersection(update_data))
        if nutrition_changed and item.nutrition_status == "VERIFIED" and "nutrition_status" not in update_data:
            raise HTTPException(
                422,
                "Updating verified nutrition requires explicitly setting its new status",
            )
        if nutrition_changed:
            update_data.update(_nutrition_data(update_data, item))
        category_id = update_data.get("category_id")
        if category_id is not None:
            category = (await session.execute(
                select(MenuCategory).where(MenuCategory.id == category_id, MenuCategory.restaurant_id == rid)
            )).scalar_one_or_none()
            if not category:
                raise HTTPException(404, "Menu category not found")
        for key, value in update_data.items():
            setattr(item, key, value)
        await session.commit()
        return {"ok": True}


@router.patch("/items/{item_id}/availability")
async def update_availability(item_id: int, body: AvailabilityUpdate, user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(MenuItem).where(MenuItem.id == item_id, MenuItem.restaurant_id == rid))
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Menu item not found")
        item.available = body.available
        await session.commit()

        # Publish real-time event
        from app.realtime.events import menu_item_availability_changed
        from app.realtime.ws_manager import ws_manager
        event = menu_item_availability_changed(item_id, body.available, rid)
        await ws_manager.broadcast(rid, event.to_dict())

        return {"ok": True, "item_id": item_id, "available": body.available}


@router.post("/items/{item_id}/images", status_code=201)
async def upload_menu_image(item_id: int, file: UploadFile = File(...), user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        item = (await session.execute(
            select(MenuItem).where(MenuItem.id == item_id, MenuItem.restaurant_id == rid)
        )).scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Menu item not found")
        file_data = await file.read()
        url = await upload_image(file_data, file.filename or "image.jpg", file.content_type or "image/jpeg")
        image = MenuItemImage(
            restaurant_id=rid,
            item_id=item_id,
            url=url,
        )
        session.add(image)
        await session.commit()
        await session.refresh(image)
        return {"id": image.id, "url": url}
