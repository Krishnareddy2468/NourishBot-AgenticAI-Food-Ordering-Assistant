import pytest
from fastapi import HTTPException

from app.api.routes import deliveries, orders, payments
from app.agent import executor
from app.agent.context_builder import ContextSnapshot


class EmptyResult:
    def scalar_one_or_none(self):
        return None


class RecordingSession:
    def __init__(self):
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        return EmptyResult()


def _compiled(statement) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True})).lower()


@pytest.mark.asyncio
async def test_order_lookup_is_scoped_to_authenticated_tenant(monkeypatch):
    session = RecordingSession()
    monkeypatch.setattr(orders, "AsyncSessionLocal", lambda: session)

    with pytest.raises(HTTPException) as exc:
        await orders.get_order(15, user={"restaurant_id": 42})

    assert exc.value.status_code == 404
    query = _compiled(session.statements[0])
    assert "orders.restaurant_id = 42" in query


@pytest.mark.asyncio
async def test_staff_delivery_tracking_is_scoped_to_authenticated_tenant(monkeypatch):
    session = RecordingSession()
    monkeypatch.setattr(deliveries, "AsyncSessionLocal", lambda: session)

    with pytest.raises(HTTPException) as exc:
        await deliveries.track_delivery(9, user={"restaurant_id": 73})

    assert exc.value.status_code == 404
    query = _compiled(session.statements[0])
    assert "deliveries.restaurant_id = 73" in query


@pytest.mark.asyncio
async def test_payment_intent_cannot_resolve_another_tenants_order(monkeypatch):
    session = RecordingSession()
    monkeypatch.setattr(payments, "AsyncSessionLocal", lambda: session)

    with pytest.raises(HTTPException) as exc:
        await payments.create_intent(88, user={"restaurant_id": 91})

    assert exc.value.status_code == 404
    query = _compiled(session.statements[0])
    assert "orders.restaurant_id = 91" in query


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["track_order", "cancel_order"])
async def test_customer_order_tools_scope_reference_to_tenant_and_owner(monkeypatch, tool):
    session = RecordingSession()
    monkeypatch.setattr(executor, "AsyncSessionLocal", lambda: session)
    ctx = ContextSnapshot(chat_id="customer-chat", restaurant_id=37, customer_id=412)

    result = await executor.Executor.run(
        [{"tool": tool, "args": {"order_ref": "NB-ABC123"}}],
        ctx,
    )

    assert result.rejected
    query = _compiled(session.statements[0])
    assert "orders.restaurant_id = 37" in query
    assert "orders.customer_id = 412" in query


@pytest.mark.asyncio
async def test_customer_chat_rejects_staff_mutation_tools_before_database_access(monkeypatch):
    def fail_if_opened():
        raise AssertionError("database should not be opened for a forbidden customer tool")

    monkeypatch.setattr(executor, "AsyncSessionLocal", fail_if_opened)
    ctx = ContextSnapshot(chat_id="customer-chat", restaurant_id=1, customer_id=10)

    result = await executor.Executor.run(
        [{"tool": "set_item_availability", "args": {"item_id": 99, "available": False}}],
        ctx,
    )

    assert not result.committed
    assert result.rejected[0].error == "This operation is not available in customer chat."
