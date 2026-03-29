"""Tests for JSON serialization/deserialization of replay responses."""

import json

from pydantic import BaseModel

from stateloom.replay.schema import (
    SESSION_SCHEMA_VERSION,
    SessionRecord,
    StepRecordSchema,
    deserialize_response,
    serialize_response,
)


class MockPydanticResponse(BaseModel):
    """Mock Pydantic response for testing serialization."""

    id: str = "resp-123"
    model: str = "gpt-4o-mini"
    content: str = "Hello, world!"


class TestSerializeResponse:
    def test_serialize_none(self):
        result = serialize_response(None)
        assert json.loads(result) is None

    def test_serialize_dict(self):
        data = {"id": "resp-123", "content": "hello"}
        result = serialize_response(data)
        parsed = json.loads(result)
        assert parsed["id"] == "resp-123"
        assert parsed["content"] == "hello"

    def test_serialize_pydantic_model(self):
        resp = MockPydanticResponse()
        result = serialize_response(resp)
        parsed = json.loads(result)
        assert parsed["id"] == "resp-123"
        assert parsed["model"] == "gpt-4o-mini"

    def test_serialize_string(self):
        result = serialize_response("just a string")
        assert json.loads(result) == "just a string"

    def test_serialize_list(self):
        result = serialize_response([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]


class TestDeserializeResponse:
    def test_deserialize_none_input(self):
        result = deserialize_response(None)
        assert result is None

    def test_deserialize_dict(self):
        json_str = json.dumps({"id": "resp-123"})
        result = deserialize_response(json_str)
        assert result == {"id": "resp-123"}

    def test_deserialize_with_provider(self):
        json_str = json.dumps({"model": "gpt-4o-mini"})
        result = deserialize_response(json_str, provider="openai")
        assert result == {"model": "gpt-4o-mini"}

    def test_roundtrip_dict(self):
        original = {"id": "resp-123", "content": "hello", "cost": 0.001}
        serialized = serialize_response(original)
        deserialized = deserialize_response(serialized)
        assert deserialized == original

    def test_roundtrip_pydantic(self):
        original = MockPydanticResponse()
        serialized = serialize_response(original)
        deserialized = deserialize_response(serialized)
        assert deserialized["id"] == "resp-123"
        assert deserialized["model"] == "gpt-4o-mini"


class TestSessionRecord:
    def test_default_version(self):
        record = SessionRecord(session_id="test")
        assert record.version == SESSION_SCHEMA_VERSION

    def test_with_steps(self):
        record = SessionRecord(
            session_id="test",
            steps=[
                StepRecordSchema(
                    step=1, event_type="llm_call", cached_response_json='{"ok": true}'
                ),
                StepRecordSchema(step=2, event_type="tool_call", tool_name="search"),
            ],
        )
        assert len(record.steps) == 2
        assert record.steps[0].cached_response_json == '{"ok": true}'
        assert record.steps[1].tool_name == "search"

    def test_serialization_roundtrip(self):
        record = SessionRecord(
            session_id="test",
            steps=[
                StepRecordSchema(step=1, event_type="llm_call"),
            ],
        )
        json_str = record.model_dump_json()
        restored = SessionRecord.model_validate_json(json_str)
        assert restored.session_id == "test"
        assert restored.version == SESSION_SCHEMA_VERSION
        assert len(restored.steps) == 1
