from app.agent.mcp_agent import _is_confirmation_intent, _is_transaction_tool, _scheduled_safe_tool


class Schema:
    def __init__(self, properties):
        self.properties = properties

    def model_json_schema(self):
        return {"type": "object", "properties": self.properties}


class Tool:
    def __init__(self, name, properties=None):
        self.name = name
        self.args_schema = Schema(properties or {})


def test_blocks_transaction_tool_without_schedule_field():
    assert _scheduled_safe_tool(Tool("place_order", {"cart_id": {"type": "string"}})) is False


def test_allows_transaction_tool_with_native_schedule_field():
    tool = Tool("checkout", {"scheduled_for": {"type": "string", "format": "date-time"}})
    assert _scheduled_safe_tool(tool) is True


def test_allows_non_transactional_discovery_tool():
    assert _scheduled_safe_tool(Tool("search_restaurants", {"query": {"type": "string"}})) is True


def test_transaction_tools_are_identified_for_discovery_mode_filtering():
    assert _is_transaction_tool("zomato__checkout") is True
    assert _is_transaction_tool("swiggy__search_restaurants") is False


def test_external_confirmation_does_not_match_embedded_yes_words():
    assert _is_confirmation_intent("yes") is True
    assert _is_confirmation_intent("yes, but use another address") is False
    assert _is_confirmation_intent("yesterday") is False
