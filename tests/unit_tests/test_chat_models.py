"""Unit tests for ChatOptiQ that need no network."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_optiq import ChatOptiQ
from langchain_optiq.chat_models import DEFAULT_API_BASE, DEFAULT_MODEL


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPTIQ_API_KEY", "OPTIQ_CLOUD_API_KEY", "OPTIQ_API_BASE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIQ_API_KEY", "sk-optiq-cloud-a")
    llm = ChatOptiQ()
    assert llm.model_name == DEFAULT_MODEL
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.api_key is not None and llm.api_key.get_secret_value() == "sk-optiq-cloud-a"
    assert str(llm.root_client.base_url).rstrip("/") == DEFAULT_API_BASE
    assert llm._llm_type == "chat-optiq"


def test_cloud_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIQ_CLOUD_API_KEY", "sk-optiq-cloud-b")
    llm = ChatOptiQ()
    assert llm.api_key is not None and llm.api_key.get_secret_value() == "sk-optiq-cloud-b"


def test_primary_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIQ_API_KEY", "sk-optiq-cloud-a")
    monkeypatch.setenv("OPTIQ_CLOUD_API_KEY", "sk-optiq-cloud-b")
    llm = ChatOptiQ()
    assert llm.api_key is not None and llm.api_key.get_secret_value() == "sk-optiq-cloud-a"


def test_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="OPTIQ_API_KEY"):
        ChatOptiQ()


def test_openai_key_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(ValueError, match="OPTIQ_API_KEY"):
        ChatOptiQ()


def test_reasoning_effort_in_payload() -> None:
    llm = ChatOptiQ(api_key="k", model="deepseek-v4.1-flash", reasoning_effort="none")
    payload = llm._get_request_payload([HumanMessage("hi")])
    assert payload["model"] == "deepseek-v4.1-flash"
    assert payload["reasoning_effort"] == "none"
    assert "messages" in payload and "input" not in payload


def test_reasoning_effort_omitted_by_default() -> None:
    llm = ChatOptiQ(api_key="k")
    payload = llm._get_request_payload([HumanMessage("hi")])
    assert "reasoning_effort" not in payload


def test_never_uses_responses_api() -> None:
    llm = ChatOptiQ(api_key="k", model="glm-5.3", reasoning_effort="low")
    assert llm._use_responses_api({"reasoning_effort": "low"}) is False


def _response(**message_extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "glm-5.3-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "4", **message_extra},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_text_kept(field: str) -> None:
    llm = ChatOptiQ(api_key="k")
    result = llm._create_chat_result(_response(**{field: "2 plus 2"}))
    message = result.generations[0].message
    assert message.content == "4"
    assert message.additional_kwargs["reasoning_content"] == "2 plus 2"
    assert message.response_metadata["model_provider"] == "optiq"


def test_no_reasoning_text() -> None:
    llm = ChatOptiQ(api_key="k")
    message = llm._create_chat_result(_response()).generations[0].message
    assert "reasoning_content" not in message.additional_kwargs


def test_stream_chunk_reasoning() -> None:
    llm = ChatOptiQ(api_key="k")
    chunk = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "glm-5.3-flash",
        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "think"}}],
    }
    gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
    assert gen is not None
    assert gen.message.additional_kwargs["reasoning_content"] == "think"
    assert gen.message.response_metadata["model_provider"] == "optiq"


def test_ls_provider() -> None:
    llm = ChatOptiQ(api_key="k")
    assert llm._get_ls_params()["ls_provider"] == "optiq"
