from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.models.schemas import ChatMessage, ChatResponse, VoiceChatMessage
from app.services.langgraph_agent import langgraph_food_agent
from app.services.session_service import session_service
from app.services.restaurant_service import restaurant_service
from app.services.zomato_mcp import global_zomato_mcp
from app.services.voice_service import voice_service
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])
CHAT_REQUEST_TIMEOUT_SECONDS = 28


@router.post("/message", response_model=ChatResponse)
async def send_message(chat: ChatMessage):
    """main chat endpoint - processes user message and returns bot response"""
    try:
        response, thinking_steps = await asyncio.wait_for(
            langgraph_food_agent.process_message(
                user_id=chat.user_id,
                message=chat.message,
                user_name=chat.user_name,
                user_location=chat.user_location,
                filters=chat.filters,
                channel="web",
                input_mode="text",
            ),
            timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
        )

        session = session_service.get_session(chat.user_id)

        # grab restaurant info if one is selected
        restaurant_data = None
        if session.selected_restaurant_id:
            rest = restaurant_service.get_restaurant_by_id(session.selected_restaurant_id)
            if rest:
                restaurant_data = {
                    "id": rest["id"],
                    "name": rest["name"],
                    "image": rest["image"],
                    "rating": rest["rating"],
                    "delivery_time": rest["delivery_time"],
                }

        order_data = None
        if session.current_order_id:
            order_data = {
                "order_id": session.current_order_id,
                "status": "placed",
            }

        return ChatResponse(
            response=response,
            state=session.state.value,
            cart_items=[item.model_dump() for item in session.cart],
            restaurant=restaurant_data,
            order=order_data,
            thinking_steps=thinking_steps,
        )

    except asyncio.TimeoutError:
        logger.error("Chat request timed out after %ss", CHAT_REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="The assistant took too long to respond. Please try a more specific request.",
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Persist sessions to disk so they survive backend restarts
        session_service.save()


@router.post("/voice-message", response_model=ChatResponse)
async def send_voice_message(chat: VoiceChatMessage):
    """Voice chat endpoint when transcript is provided by client."""
    try:
        response, thinking_steps = await asyncio.wait_for(
            langgraph_food_agent.process_message(
                user_id=chat.user_id,
                message=chat.transcript,
                user_name=chat.user_name,
                user_location=chat.user_location,
                filters=chat.filters,
                channel="web",
                input_mode="voice",
            ),
            timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
        )
        session = session_service.get_session(chat.user_id)
        return ChatResponse(
            response=response,
            state=session.state.value,
            cart_items=[item.model_dump() for item in session.cart],
            thinking_steps=thinking_steps,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Voice request timed out. Please retry.")
    finally:
        session_service.save()


@router.post("/voice-upload", response_model=ChatResponse)
async def send_voice_upload(
    audio: UploadFile = File(...),
    user_id: str = Form("web_user"),
    user_name: str = Form("Guest"),
    user_location: str | None = Form(None),
):
    """Voice chat endpoint for raw uploaded audio."""
    try:
        audio_bytes = await audio.read()
        transcript = await voice_service.transcribe_ogg(audio_bytes)
        response, thinking_steps = await asyncio.wait_for(
            langgraph_food_agent.process_message(
                user_id=user_id,
                message=transcript,
                user_name=user_name,
                user_location=user_location,
                channel="web",
                input_mode="voice",
            ),
            timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
        )
        session = session_service.get_session(user_id)
        return ChatResponse(
            response=response,
            state=session.state.value,
            cart_items=[item.model_dump() for item in session.cart],
            thinking_steps=["Transcribed voice input", *thinking_steps],
        )
    except Exception as e:
        logger.error("Voice upload processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session_service.save()


@router.get("/session/{user_id}")
async def get_session(user_id: str):
    session = session_service.get_session(user_id)
    restaurant_data = None
    if session.selected_restaurant_id:
        rest = restaurant_service.get_restaurant_by_id(session.selected_restaurant_id)
        if rest:
            restaurant_data = {
                "id": rest["id"],
                "name": rest["name"],
                "image": rest["image"],
                "rating": rest["rating"],
            }

    return {
        "user_id": session.user_id,
        "state": session.state.value,
        "restaurant": restaurant_data,
        "cart_items": [item.model_dump() for item in session.cart],
        "current_order_id": session.current_order_id,
    }


@router.post("/reset/{user_id}")
async def reset_session(user_id: str):
    session_service.reset_session(user_id)
    session_service.save()
    return {"message": "Session reset successfully"}


@router.get("/order-status/{user_id}")
async def get_order_status(user_id: str):
    """Lightweight polling endpoint for live order status via Zomato MCP."""
    session = session_service.get_session(user_id)
    if not session.current_order_id:
        return {"order_id": None, "status": None, "message": None}

    try:
        result = await global_zomato_mcp.call_tool("get_order_tracking_info", {})
        for chunk in result or []:
            if isinstance(chunk, str):
                try:
                    data = json.loads(chunk)
                    if isinstance(data, dict) and data.get("error_message"):
                        return {
                            "order_id": session.current_order_id,
                            "status": "unknown",
                            "message": data.get("error_message"),
                        }
                    # Walk nested structure: data.order_tracking.order_tracking_items
                    orders = []
                    if isinstance(data, dict):
                        ot = data.get("order_tracking")
                        if isinstance(ot, dict):
                            orders = ot.get("order_tracking_items", [])
                        if not orders:
                            for key in ("orders", "active_orders", "order_tracking_items"):
                                items = data.get(key)
                                if isinstance(items, list) and items:
                                    orders = items
                                    break
                        if not orders and (data.get("order_id") or data.get("order_status")):
                            orders = [data]
                    elif isinstance(data, list):
                        orders = data

                    if orders and isinstance(orders, list):
                        order = orders[0]
                        rider = order.get("rider") or {}
                        return {
                            "order_id": order.get("order_id") or order.get("id") or session.current_order_id,
                            "status": order.get("order_status") or order.get("status") or "placed",
                            "message": order.get("message") or order.get("order_status") or "Order placed",
                            "restaurant_name": order.get("restaurant_name", ""),
                            "delivery_partner": (rider.get("name") or rider.get("rider_name")) if isinstance(rider, dict) else None,
                        }
                except Exception:
                    pass
    except Exception as e:
        logger.error("Order status polling failed: %s", e)

    return {
        "order_id": session.current_order_id,
        "status": "placed",
        "message": "Order placed on Zomato",
    }
