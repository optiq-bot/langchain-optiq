"""LangChain standard unit tests for ChatOptiQ."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_optiq import ChatOptiQ


class TestChatOptiQUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOptiQ]:
        return ChatOptiQ

    @property
    def chat_model_params(self) -> dict:
        return {"model": "glm-5.3-flash", "api_key": "sk-optiq-cloud-test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "OPTIQ_API_KEY": "sk-optiq-cloud-env",
                "OPTIQ_API_BASE": "https://example.invalid/api/v1",
            },
            {"model": "glm-5.3-flash"},
            {
                "api_key": "sk-optiq-cloud-env",
                "api_base": "https://example.invalid/api/v1",
            },
        )

    @property
    def supports_image_inputs(self) -> bool:
        return False
