import asyncio
import types
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.providers.openrouter import OpenRouterProvider
from app.models.providers import OpenRouterConfig
from app.core.providers.base import ProviderMaxToolIterationsError, ProviderConnectionError


@pytest.fixture
def openrouter_config():
    return OpenRouterConfig(
        name="OpenRouter",
        provider_type=openrouter_provider_type(),  # helper returns ProviderType.OPENROUTER without importing enum directly
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/auto",
        model_list=["openrouter/auto"],
        api_key="sk-or-test"
    )


def openrouter_provider_type():
    # Avoid direct enum import issues in tests if file layout changes
    from app.models.providers import ProviderType
    return ProviderType.OPENROUTER


@pytest.fixture
def provider(openrouter_config):
    return OpenRouterProvider(openrouter_config)


def _make_chat_completion_message(content=None, tool_calls=None):
    # Build an object similar to OpenAI SDK response shapes used by providers
    # response.choices[0].message.content and .tool_calls
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or []
    )
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])
    return response


def _make_tool_call(id_, name, args_json):
    fn = SimpleNamespace(name=name, arguments=args_json)
    tc = SimpleNamespace(id=id_, function=fn)
    return tc


async def _async_stream_generator(chunks):
    for ch in chunks:
        yield ch


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls
    )
    choice = SimpleNamespace(
        delta=delta,
        finish_reason=finish_reason
    )
    chunk = SimpleNamespace(choices=[choice])
    return chunk


@pytest.mark.asyncio
async def test_initialize_sets_client_and_model_list(provider, monkeypatch):
    mocked_client = MagicMock()
    mocked_models = SimpleNamespace(data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")])
    mocked_client.models.list = AsyncMock(return_value=mocked_models)

    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    assert provider.client is not None
    # Best-effort update of model_list
    assert "model-a" in provider.config.model_list
    assert "model-b" in provider.config.model_list


@pytest.mark.asyncio
async def test_initialize_missing_api_key_logs_and_keeps_client_none(monkeypatch):
    cfg = OpenRouterConfig(
        name="OpenRouter",
        provider_type=openrouter_provider_type(),
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/auto",
        model_list=["openrouter/auto"],
        api_key=""  # missing
    )
    prov = OpenRouterProvider(cfg)
    # initialize catches and logs warning; no exception should propagate
    await prov.initialize()
    # Client remains None due to missing key
    assert prov.client is None


@pytest.mark.asyncio
async def test_health_check_healthy(provider, monkeypatch):
    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock(return_value=_make_chat_completion_message(content="ok"))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()
    health = await provider.health_check()
    assert health.status == "healthy"


@pytest.mark.asyncio
async def test_health_check_unhealthy(provider, monkeypatch):
    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock(side_effect=Exception("bad"))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()
    health = await provider.health_check()
    assert health.status == "unhealthy"
    assert "bad" in (health.error_details or "")


@pytest.mark.asyncio
async def test_get_model_list(provider):
    assert provider.get_model_list.__name__  # simple existence
    # returns config value
    assert provider.config.model_list == await provider.get_model_list()


@pytest.mark.asyncio
async def test_send_chat_returns_content_without_tools(provider, monkeypatch):
    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock(
        return_value=_make_chat_completion_message(content="hello world")
    )
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    context = [{"role": "user", "content": "hi"}]
    out = await provider.send_chat(context, provider.config.default_model, "You are helpful.")
    assert out == "hello world"
    # one successful call recorded
    assert provider.total_requests == 1
    assert provider.success_requests == 1


@pytest.mark.asyncio
async def test_send_chat_executes_tool_calls_and_returns_final(provider, monkeypatch):
    # First response includes tool call
    tc = _make_tool_call("call-1", "add_two_numbers", '{"a":2,"b":3}')
    first = _make_chat_completion_message(content=None, tool_calls=[tc])
    # Second response returns final content without tool calls
    second = _make_chat_completion_message(content="sum is 5", tool_calls=[])

    mocked_client = MagicMock()
    # The provider will call create twice in the loop
    mocked_client.chat.completions.create = AsyncMock(side_effect=[first, second])

    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    # Patch execute_tool_call to return tool result
    with patch.object(provider, "execute_tool_call", new=AsyncMock(return_value="5")) as exec_mock:
        context = [{"role": "user", "content": "calc 2+3"}]
        result = await provider.send_chat(context, provider.config.default_model, "You are helpful.")
        assert result == "sum is 5"
        exec_mock.assert_awaited_once()
        # Verify correct tool name and parsed args
        called_args = exec_mock.call_args.kwargs if exec_mock.call_args.kwargs else {}
        # Python 3.8/3.9: call_args is tuple (args, kwargs)
        if not called_args:
            # fallback to args tuple: (tool_name, arguments, agent_id)
            args_tuple = exec_mock.call_args.args
            assert args_tuple[0] == "add_two_numbers"
            assert args_tuple[1] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_send_chat_max_tool_iterations_raises(provider, monkeypatch):
    # Every response keeps asking for a tool to force iteration
    tc = _make_tool_call("call-1", "noop", "{}")
    looping = _make_chat_completion_message(content=None, tool_calls=[tc])

    mocked_client = MagicMock()
    # Enough responses to exceed default max_tool_iterations (10) -> we set small for speed
    mocked_client.chat.completions.create = AsyncMock(side_effect=[looping] * 5)

    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    provider.max_tool_iterations = 2
    with patch.object(provider, "execute_tool_call", new=AsyncMock(return_value="ok")):
        with pytest.raises(ProviderMaxToolIterationsError):
            await provider.send_chat([{"role": "user", "content": "loop"}], provider.config.default_model, "sys")


@pytest.mark.asyncio
async def test_send_chat_with_streaming_yields_content_and_executes_tools(provider, monkeypatch):
    # Streaming chunks: yield "He" then "llo", then a tool call in fragments, finish_reason=tool_calls
    tool_tc_part1 = SimpleNamespace(id="id-1", function=SimpleNamespace(name="add", arguments='{"a":'))
    # Provide valid concatenation that results in {"a":1,"b":2}
    tool_tc_part2 = SimpleNamespace(id="id-1", function=SimpleNamespace(name=None, arguments='1,"b":2}'))

    chunks = [
        _make_stream_chunk(content="He", tool_calls=None, finish_reason=None),
        _make_stream_chunk(content="llo", tool_calls=None, finish_reason=None),
        _make_stream_chunk(content=None, tool_calls=[tool_tc_part1], finish_reason=None),
        _make_stream_chunk(content=None, tool_calls=[tool_tc_part2], finish_reason="tool_calls"),
    ]

    # After tools executed, a follow-up non-streaming call returns final text w/o tool calls
    final_after_tools = _make_chat_completion_message(content="done", tool_calls=[])

    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock()
    # First call (streaming) returns async generator
    mocked_client.chat.completions.create.side_effect = [
        _async_stream_generator(chunks),  # streaming
        final_after_tools                  # follow-up after tool execution
    ]

    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    # Patch execute_tool_call to ensure it's called once with concatenated args
    async def _exec_tool(name, args, agent_id):
        assert name == "add"
        # concatenated JSON should be valid: {"a":1,"b":2}
        assert isinstance(args, dict)
        assert args.get("a") == 1 and args.get("b") == 2
        return "ok"

    with patch.object(provider, "execute_tool_call", new=AsyncMock(side_effect=_exec_tool)) as exec_mock:
        out_stream = provider.send_chat_with_streaming(
            [{"role": "user", "content": "stream"}],
            provider.config.default_model,
            "sys"
        )
        collected = []
        async for piece in out_stream:
            collected.append(piece)

        # Should have yielded the two content chunks
        assert "".join(collected[:2]) == "Hello"
        # The rest may include nothing or warnings; ensure at least we got two pieces
        assert len(collected) >= 2
        # Ensure tool execution occurred exactly once in this flow
        assert getattr(exec_mock, "await_count", 0) == 1


@pytest.mark.asyncio
async def test_parse_streaming_tool_calls_concatenates_arguments(provider, monkeypatch):
    # Directly exercise _parse_streaming_tool_calls to ensure concatenation works
    chunks = [
        _make_stream_chunk(content=None, tool_calls=[SimpleNamespace(id="t1", function=SimpleNamespace(name="fn", arguments='{"x":'))]),
        _make_stream_chunk(content=None, tool_calls=[SimpleNamespace(id="t1", function=SimpleNamespace(name=None, arguments='1,"y":2}'))], finish_reason="tool_calls"),
    ]
    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock(return_value=_async_stream_generator(chunks))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    # Run through parser
    gen = provider._parse_streaming_tool_calls(await mocked_client.chat.completions.create())
    content = ""
    tool_calls = None
    async for kind, data, calls in gen:
        if kind == "content":
            content += data
        if kind == "final":
            tool_calls = calls

    assert tool_calls is not None
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["name"] == "fn"
    assert tc["args"] == '{"x":1,"y":2}'


@pytest.mark.asyncio
async def test_embed_returns_vector(provider, monkeypatch):
    mocked_client = MagicMock()
    mocked_client.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    vec = await provider.embed("hello", model="openrouter/embedding")
    assert vec == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_missing_fields_raise(provider, monkeypatch):
    mocked_client = MagicMock()
    # Missing data
    mocked_client.embeddings.create = AsyncMock(return_value=SimpleNamespace())
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()
    with pytest.raises(AttributeError):
        await provider.embed("hello", model="openrouter/embedding")

    # Empty data
    mocked_client.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[]))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()
    with pytest.raises(IndexError):
        await provider.embed("hello", model="openrouter/embedding")

    # Missing embedding
    mocked_client.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace()]))
    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()
    with pytest.raises(AttributeError):
        await provider.embed("hello", model="openrouter/embedding")


@pytest.mark.asyncio
async def test_streaming_error_yields_error_and_terminates(provider, monkeypatch):
    async def _bad_stream():
        raise RuntimeError("streaming broke")

    mocked_client = MagicMock()
    mocked_client.chat.completions.create = AsyncMock(return_value=_bad_stream())

    with patch("app.core.providers.openrouter.AsyncOpenAI", return_value=mocked_client):
        await provider.initialize()

    gen = provider.send_chat_with_streaming([{"role": "user", "content": "x"}], provider.config.default_model, "sys")
    pieces = []
    async for p in gen:
        pieces.append(p)

    # At least one error string is yielded
    assert any("Error: Streaming interrupted" in p for p in pieces) or any("Error:" in p for p in pieces)


def test_provider_manager_has_openrouter_entry():
    # Avoid any external patching leaking in; import fresh
    from importlib import reload
    import app.core.providers.manager as mgr_mod
    pm = reload(mgr_mod).ProviderManager()
    providers = pm.list_providers()
    # Ensure provider is registered
    assert isinstance(providers, dict)
    assert "openrouter" in providers
    entry = providers["openrouter"]
    assert entry["name"] == "OpenRouter"
    # Validate config and class bindings strictly
    from app.models.providers import OpenRouterConfig
    from app.core.providers.openrouter import OpenRouterProvider
    assert entry["config_class"] is OpenRouterConfig
    assert entry["class"] is OpenRouterProvider