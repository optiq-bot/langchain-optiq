"""LangChain standard integration tests for ChatOptiQ. Needs OPTIQ_API_KEY."""

from __future__ import annotations

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_optiq import ChatOptiQ


class TestChatOptiQIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatOptiQ]:
        return ChatOptiQ

    @property
    def chat_model_params(self) -> dict:
        return {"model": "deepseek-v4.1-flash", "reasoning_effort": "none", "temperature": 0}

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return True
