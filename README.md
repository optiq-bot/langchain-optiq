# langchain-optiq

LangChain chat model for [OptiQ Cloud](https://mlx-optiq.com/docs/cloud/api).
`ChatOptiQ` is a `ChatOpenAI`-style model with the OptiQ Cloud base URL, key and
model ids set, plus `reasoning_effort` passed through to the API.

## Installation

```bash
pip install -U langchain-optiq
export OPTIQ_API_KEY="sk-optiq-cloud-..."
```

Create a key on the OptiQ Cloud dashboard. `OPTIQ_CLOUD_API_KEY`, the name used in the
OptiQ docs, is read when `OPTIQ_API_KEY` is not set.

## Usage

```python
from langchain_optiq import ChatOptiQ

llm = ChatOptiQ(model="deepseek-v4.1-flash", reasoning_effort="low")
print(llm.invoke("Name three uses of a hash map.").content)
```

Streaming, async, tool calling and structured output work as they do with `ChatOpenAI`:

```python
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str


llm = ChatOptiQ(model="glm-5.3-flash", reasoning_effort="minimal")
llm.with_structured_output(City).invoke("The capital of Japan")

for chunk in llm.stream("Write a haiku about caches."):
    print(chunk.content, end="")
```

## Models

| model | context | reasoning_effort values |
|---|---|---|
| `glm-5.3-flash` (default) | 1,048,576 | `minimal`, `low`, `medium`, `high`, `xhigh` |
| `glm-5.3` | 1,048,576 | `minimal`, `low`, `medium`, `high`, `xhigh` |
| `deepseek-v4.1-flash` | 1,040,000 | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `minimax-m3.1-flash` | 1,000,000 | `minimal`, `low`, `medium`, `high`, `xhigh` |

The current list, with prices, is at `GET https://cloud.mlx-optiq.com/api/v1/models`
(no key needed). Any model id from that list can be passed as `model`.

## Reasoning

Every model thinks before it answers, and thinking is billed as output. With no
`reasoning_effort`, a model thinks fully. `minimal` or `low` thinks least on the models
that always think. Only `deepseek-v4.1-flash` takes `none`, which turns thinking off. A
value the model does not take comes back as a 400 before the model runs, so nothing is
charged. The measured effect of each value is in the
[API reference](https://mlx-optiq.com/docs/cloud/api#reasoning).

When the API returns the thinking text, it is kept in
`message.additional_kwargs["reasoning_content"]`.

## Configuration

| argument | environment variable | default |
|---|---|---|
| `api_key` | `OPTIQ_API_KEY`, then `OPTIQ_CLOUD_API_KEY` | (required) |
| `base_url` | `OPTIQ_API_BASE` | `https://cloud.mlx-optiq.com/api/v1` |
| `model` | | `glm-5.3-flash` |

Other `ChatOpenAI` arguments such as `max_tokens`, `temperature`, `timeout`,
`max_retries` and `default_headers` are passed through. Sending the same `x-session-id`
header on every call of one task keeps that task's prompt cache:

```python
llm = ChatOptiQ(default_headers={"x-session-id": "task-42"})
```

## Development

```bash
uv sync --all-groups
make test               # unit tests, network disabled
make integration_tests  # live API, needs OPTIQ_API_KEY
make lint
```

## Links

- API reference: [mlx-optiq.com/docs/cloud/api](https://mlx-optiq.com/docs/cloud/api)
- Other frameworks: [mlx-optiq.com/docs/cloud/frameworks](https://mlx-optiq.com/docs/cloud/frameworks)

## License

MIT
