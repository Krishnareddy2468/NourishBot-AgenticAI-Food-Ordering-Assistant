"""
Menu routes — CRUD for menu items, categories, images, modifiers.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models import MenuCategory, MenuItem, MenuItemImage
from app.auth.deps import require_manager
from app.core.storage import upload_image

router = APIRouter(prefix="/api/v1/menu", tags=["menu"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class MenuItemCreate(BaseModel):
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


class MenuItemUpdate(BaseModel):
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


class AvailabilityUpdate(BaseModel):
    available: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/items")
async def list_menu_items():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MenuItem).order_by(MenuItem.name)
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
                }
                for i in items
            ]
        }


@router.post("/items", status_code=201)
async def create_menu_item(body: MenuItemCreate, user=Depends(require_manager)):
    async with AsyncSessionLocal() as session:
        item = MenuItem(
            restaurant_id=user.get("restaurant_id", 1),
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
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return {"id": item.id, "name": item.name}


@router.patch("/items/{item_id}")
async def update_menu_item(item_id: int, body: MenuItemUpdate, user=Depends(require_manager)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(MenuItem).where(MenuItem.id == item_id))
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Menu item not found")

        update_data = body.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(item, key, value)
        await session.commit()
        return {"ok": True}


@router.patch("/items/{item_id}/availability")
async def update_availability(item_id: int, body: AvailabilityUpdate, user=Depends(require_manager)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(MenuItem).where(MenuItem.id == item_id))
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Menu item not found")
        item.available = body.available
        await session.commit()

        # Publish real-time event
        from app.realtime.events import menu_item_availability_changed
        from app.realtime.ws_manager import ws_manager
        event = menu_item_availability_changed(item_id, body.available)
        await ws_manager.broadcast(1, event.to_dict())

        return {"ok": True, "item_id": item_id, "available": body.available}


@router.post("/items/{item_id}/images", status_code=201)
async def upload_menu_image(item_id: int, file: UploadFile = File(...), user=Depends(require_manager)):
    file_data = await file.read()
    url = await upload_image(file_data, file.filename or "image.jpg", file.content_type or "image/jpeg")

    async with AsyncSessionLocal() as session:
        image = MenuItemImage(
            restaurant_id=user.get("restaurant_id", 1),
            item_id=item_id,
            url=url,
        )
        session.add(image)
        await session.commit()
        await session.refresh(image)
        return {"id": image.id, "url": url}
