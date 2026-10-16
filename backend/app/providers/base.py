"""Provider adapter contract and capability vocabulary."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class CapabilityState(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"


class ProviderMode(StrEnum):
    DISCOVERY = "DISCOVERY"
    HANDOFF = "HANDOFF"
    TRANSACTIONAL = "TRANSACTIONAL"


@dataclass(frozen=True)
class ProviderCapabilities:
    discovery: bool = False
    menu_lookup: bool = False
    live_price_verification: bool = False
    scheduled_ordering: bool = False
    scheduling_horizon_days: int | None = None
    delivery: bool = False
    pickup: bool = False
    curbside_drive_through: bool = False
    cancellation: bool = False
    order_status_lookup: bool = False
    user_owned_authentication: bool = False
    stable_idempotency: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ProviderCapabilities":
        source = value or {}
        return cls(**{field: source.get(field) for field in cls.__dataclass_fields__ if field in source})


@dataclass(frozen=True)
class ProviderPolicy:
    provider: str
    version_id: int | None
    version: int
    state: CapabilityState
    mode: ProviderMode
    capabilities: ProviderCapabilities
    tool_schema_hash: str | None = None
    adapter_version: str = "mcp-v1"

    @property
    def allows_transactional_mode(self) -> bool:
        caps = self.capabilities
        return (
            self.state == CapabilityState.VERIFIED
            and self.mode == ProviderMode.TRANSACTIONAL
            and bool(self.tool_schema_hash)
            and caps.scheduled_ordering
            and caps.order_status_lookup
            and caps.stable_idempotency
        )


@dataclass(frozen=True)
class AdapterResult:
    outcome: str
    response: dict[str, Any]
    provider_order_id: str | None = None
    provider_request_id: str | None = None


class ProviderAdapter(ABC):
    """The only interface allowed to perform provider-specific operations."""

    @abstractmethod
    async def discover(self, query: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    async def create_handoff(self, context: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    async def submit_scheduled_order(self, request: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    async def get_order_status(self, reference: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, reference: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError
