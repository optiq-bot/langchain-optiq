"""One live call against OptiQ Cloud. Needs OPTIQ_API_KEY."""

from langchain_optiq import ChatOptiQ


def test_invoke() -> None:
    llm = ChatOptiQ(model="deepseek-v4.1-flash", reasoning_effort="none", max_tokens=20)
    reply = llm.invoke("Reply with the single word: ready")
    assert isinstance(reply.content, str) and reply.content.strip()
    assert reply.usage_metadata is not None and reply.usage_metadata["total_tokens"] > 0
    assert reply.response_metadata["model_provider"] == "optiq"
