"""OptiQ Cloud chat models."""

from __future__ import annotations

import os
from typing import Any

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_optiq._version import __version__

DEFAULT_API_BASE = "https://cloud.mlx-optiq.com/api/v1"
DEFAULT_MODEL = "glm-5.3-flash"
API_KEY_ENV_VARS = ("OPTIQ_API_KEY", "OPTIQ_CLOUD_API_KEY")


def _api_key_from_env() -> SecretStr | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return SecretStr(value)
    return None


class ChatOptiQ(BaseChatOpenAI):
    """Chat model for OptiQ Cloud.

    OptiQ Cloud serves open models over OpenAI's chat completions format at
    `https://cloud.mlx-optiq.com/api/v1`. The model ids are listed by
    `GET https://cloud.mlx-optiq.com/api/v1/models`, which needs no key.

    Setup:
        Install `langchain-optiq` and set `OPTIQ_API_KEY` to a key from the
        OptiQ Cloud dashboard (`OPTIQ_CLOUD_API_KEY` is read as a fallback).

        ```bash
        pip install -U langchain-optiq
        export OPTIQ_API_KEY="sk-optiq-cloud-..."
        ```

    Key init args:
        model: Model id, for example `glm-5.3-flash`, `glm-5.3`,
            `deepseek-v4.1-flash` or `minimax-m3.1-flash`.
        reasoning_effort: How much the model thinks before it answers. Every
            model takes `minimal`, `low`, `medium`, `high` and `xhigh`; only
            `deepseek-v4.1-flash` also takes `none`. Left unset, the model
            thinks fully. Values a model does not take are refused with a 400.
        max_tokens: Cap on the reply, thinking included.
        temperature: Left unset, the model's tested default is used.
        api_key: OptiQ Cloud key. Read from `OPTIQ_API_KEY` if not passed.
        base_url: API base URL. Read from `OPTIQ_API_BASE` if not passed.

    Instantiate:
        ```python
        from langchain_optiq import ChatOptiQ

        model = ChatOptiQ(model="deepseek-v4.1-flash", reasoning_effort="low")
        ```

    Invoke:
        ```python
        model.invoke("Name three uses of a hash map.")
        ```

    Tool calling, structured output and streaming work as in `ChatOpenAI`.
    The text of the model's thinking, when the API returns it, is kept in
    `additional_kwargs["reasoning_content"]`.
    """

    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model id from `GET /models`."""
    api_key: SecretStr | None = Field(default_factory=_api_key_from_env)
    """OptiQ Cloud API key, read from `OPTIQ_API_KEY` if not provided."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("OPTIQ_API_BASE", default=DEFAULT_API_BASE),
    )
    """OptiQ Cloud API base URL, read from `OPTIQ_API_BASE` if not provided."""
    use_responses_api: bool | None = False
    """OptiQ Cloud serves chat completions only, so the Responses API is off."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-optiq"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "OPTIQ_API_KEY", "openai_api_key": "OPTIQ_API_KEY"}

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "optiq"
        return ls_params

    @model_validator(mode="after")
    def _set_optiq_version(self) -> Self:
        """Record the package version in the model's metadata."""
        self._add_version("langchain-optiq", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Check the key and build the OpenAI clients for OptiQ Cloud."""
        if self.api_base.rstrip("/") == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = (
                "No OptiQ Cloud API key found. Pass `api_key=...` or set the "
                "OPTIQ_API_KEY environment variable."
            )
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }
        if not self.client:
            self.root_client = openai.OpenAI(**client_params, http_client=self.http_client)
            self.client = self.root_client.chat.completions
        if not self.async_client:
            self.root_async_client = openai.AsyncOpenAI(
                **client_params, http_client=self.http_async_client
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        for generation in result.generations:
            generation.message.response_metadata["model_provider"] = "optiq"
        choices = (
            response.get("choices")
            if isinstance(response, dict)
            else getattr(response, "choices", None)
        )
        if choices and result.generations:
            message = choices[0]["message"] if isinstance(choices[0], dict) else choices[0].message
            reasoning = _reasoning_text(message)
            if reasoning:
                result.generations[0].message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or not isinstance(generation_chunk.message, AIMessageChunk):
            return generation_chunk
        generation_chunk.message.response_metadata = {
            **generation_chunk.message.response_metadata,
            "model_provider": "optiq",
        }
        if choices := chunk.get("choices"):
            reasoning = _reasoning_text(choices[0].get("delta") or {})
            if reasoning:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


def _reasoning_text(message: Any) -> str | None:
    """Read the thinking text from a message or delta, whichever field holds it."""
    for field in ("reasoning_content", "reasoning"):
        if isinstance(message, dict):
            value = message.get(field)
        else:
            value = getattr(message, field, None)
            if value is None:
                extra = getattr(message, "model_extra", None) or {}
                value = extra.get(field)
        if isinstance(value, str) and value:
            return value
    return None
