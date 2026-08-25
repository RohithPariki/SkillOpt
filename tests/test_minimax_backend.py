"""Tests for the OpenAI-compatible MiniMax chat backend."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _UrlopenRecorder:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        self.calls.append(
            {
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"content": self.content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )


class _OpenAIClientStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _install_openai_stub() -> None:
    if "openai" in sys.modules or importlib.util.find_spec("openai") is not None:
        return
    openai_stub = types.ModuleType("openai")
    openai_stub.AzureOpenAI = _OpenAIClientStub
    openai_stub.OpenAI = _OpenAIClientStub
    sys.modules["openai"] = openai_stub


@pytest.fixture()
def minimax_backend() -> Iterator[Any]:
    _install_openai_stub()
    from skillopt.model import minimax_backend as backend

    snapshot = {
        "ENABLE_THINKING": backend.ENABLE_THINKING,
        "TARGET_DEPLOYMENT": backend.TARGET_DEPLOYMENT,
        "API_KEY": backend.API_KEY,
        "BASE_URL": backend.BASE_URL,
    }
    backend.reset_token_tracker()
    yield backend
    backend.reset_token_tracker()
    for key, value in snapshot.items():
        setattr(backend, key, value)


def _record_urlopen(monkeypatch: pytest.MonkeyPatch, backend: Any) -> _UrlopenRecorder:
    recorder = _UrlopenRecorder()
    monkeypatch.setattr(backend.urllib.request, "urlopen", recorder)
    return recorder


def test_default_deployment_is_current_model(minimax_backend: Any) -> None:
    from skillopt.model.common import default_model_for_backend

    assert default_model_for_backend("minimax_chat") == "MiniMax-M3"


def test_always_on_model_sends_adaptive_not_disabled(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    """M2.x cannot turn thinking off, so never claim it is disabled.

    MiniMax documents that the M2 family accepts ``{"type": "disabled"}`` but
    keeps thinking on anyway. Sending "disabled" would record a request that
    does not match what the model actually does.
    """
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M2.7"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["thinking"] == {"type": "adaptive"}


def test_adaptive_model_respects_disabled_flag(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["model"] == "MiniMax-M3"
    assert payload["thinking"] == {"type": "disabled"}


def test_adaptive_model_respects_enabled_flag(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    minimax_backend.ENABLE_THINKING = True
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert recorder.calls[0]["payload"]["thinking"] == {"type": "adaptive"}


def test_unsupported_chat_template_kwargs_is_never_sent(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    """Guards the original regression.

    ``chat_template_kwargs.enable_thinking`` is a Qwen/HuggingFace-serving
    convention. It appears nowhere in MiniMax's OpenAI-compatible reference, so
    the endpoint ignores it -- meaning thinking silently stayed at the server
    default no matter what the flag said.
    """
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert "chat_template_kwargs" not in recorder.calls[0]["payload"]


def test_unknown_deployment_defaults_to_adaptive(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    """An unrecognized model follows the documented API default (thinking on)."""
    minimax_backend.ENABLE_THINKING = True
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-Future-9"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert recorder.calls[0]["payload"]["thinking"] == {"type": "adaptive"}


def test_optimizer_deployment_and_timeout_forwarding(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    minimax_backend.set_optimizer_deployment("MiniMax-Optimizer-Custom")
    minimax_backend.set_target_deployment("MiniMax-Target-Custom")
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    # chat_optimizer uses optimizer deployment
    minimax_backend.chat_optimizer("sys_opt", "user_opt", retries=1, timeout=45.0)
    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Optimizer-Custom"
    assert recorder.calls[0]["timeout"] == 45.0

    # chat_target uses target deployment
    minimax_backend.chat_target("sys_tgt", "user_tgt", retries=1, timeout=60.0)
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-Target-Custom"
    assert recorder.calls[1]["timeout"] == 60.0


def test_chat_optimizer_messages_dispatches_with_tools(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    minimax_backend.set_optimizer_deployment("MiniMax-Opt-Tools")
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    messages = [{"role": "user", "content": "analyze"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    minimax_backend.chat_optimizer_messages(
        messages=messages,
        retries=1,
        tools=tools,
        tool_choice="auto",
        timeout=30.0,
    )

    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Opt-Tools"
    assert recorder.calls[0]["payload"]["tools"] == tools
    assert recorder.calls[0]["payload"]["tool_choice"] == "auto"
    assert recorder.calls[0]["timeout"] == 30.0


def test_model_module_level_minimax_setters_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.model as model
    from skillopt.model import minimax_backend as backend

    recorder = _record_urlopen(monkeypatch, backend)

    model.set_target_backend("minimax_chat")
    model.set_optimizer_backend("minimax_chat")
    model.set_target_deployment("MiniMax-Global-Target")
    model.set_optimizer_deployment("MiniMax-Global-Optimizer")

    assert backend.TARGET_DEPLOYMENT == "MiniMax-Global-Target"
    assert backend.OPTIMIZER_DEPLOYMENT == "MiniMax-Global-Optimizer"

    model.chat_target("sys", "user", retries=1, timeout=99.0)
    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Global-Target"
    assert recorder.calls[0]["timeout"] == 99.0

    model.chat_optimizer_messages(
        messages=[{"role": "user", "content": "hi"}],
        retries=1,
        timeout=12.0,
    )
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-Global-Optimizer"
    assert recorder.calls[1]["timeout"] == 12.0

