"""Tests for send_reasoning_prompt.py.


Run the script by:
python -m pytest -q tests/test_reasoning_prompt.py
"""

import argparse

import pytest

import send_reasoning_prompt as module


def test_parse_args_defaults(monkeypatch):
    """CLI should expose stable defaults."""
    monkeypatch.setattr("sys.argv", ["send_reasoning_prompt.py"])
    args = module.parse_args()

    assert args.message == "gen-ai-response"
    assert args.temperature == 0.2
    assert args.max_tokens == 1024
    assert args.top_k == 40
    assert args.raw is False


def test_parse_args_raw_flag(monkeypatch):
    """--raw should flip the execution mode flag."""
    monkeypatch.setattr("sys.argv", ["send_reasoning_prompt.py", "--raw"])
    args = module.parse_args()
    assert args.raw is True


def test_main_json_mode_calls_ask_json(monkeypatch, capsys):
    """Non-raw mode should call ask_json and print raw_response."""
    args = argparse.Namespace(
        message="gen-ai-response",
        temperature=0.0,
        max_tokens=128,
        top_k=1,
        raw=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    client_double = type(
        "ClientDouble",
        (),
        {
            "ask_json": lambda self, **kwargs: {"raw_response": "ok-json"},
            "ask_text": lambda self, **kwargs: "unused",
        },
    )()
    monkeypatch.setattr(module, "AzureLLMClient", lambda: client_double)

    module.main()
    captured = capsys.readouterr().out
    assert "ok-json" in captured


def test_main_raw_mode_calls_ask_text(monkeypatch, capsys):
    """Raw mode should call ask_text and print plain response."""
    args = argparse.Namespace(
        message="gen-ai-response",
        temperature=0.0,
        max_tokens=128,
        top_k=1,
        raw=True,
        prompt="hello world",
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    client_double = type(
        "ClientDouble",
        (),
        {
            "ask_json": lambda self, **kwargs: {"raw_response": "unused"},
            "ask_text": lambda self, **kwargs: "ok-text",
        },
    )()
    monkeypatch.setattr(module, "AzureLLMClient", lambda: client_double)

    module.main()
    captured = capsys.readouterr().out
    assert "ok-text" in captured


def test_parse_args_custom_values(monkeypatch):
    """Custom CLI args should be passed through correctly."""
    monkeypatch.setattr(
        "sys.argv",
        ["send_reasoning_prompt.py", "--temperature", "0.8", "--max-tokens", "2048", "--top-k", "10"],
    )
    args = module.parse_args()
    assert args.temperature == 0.8
    assert args.max_tokens == 2048
    assert args.top_k == 10


def test_main_json_mode_passes_parameters_to_client(monkeypatch, capsys):
    """Non-raw mode should pass temperature, max_tokens, top_k to the LLM client."""
    captured_kwargs = {}
    args = argparse.Namespace(
        message="gen-ai-response",
        temperature=0.5,
        max_tokens=256,
        top_k=20,
        raw=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    def fake_ask_json(**kwargs):
        captured_kwargs.update(kwargs)
        return {"raw_response": "ok"}

    client_double = type(
        "ClientDouble", (), {"ask_json": lambda self, **kw: fake_ask_json(**kw)}
    )()
    monkeypatch.setattr(module, "AzureLLMClient", lambda: client_double)
    module.main()
    assert "parameters" in captured_kwargs
    assert captured_kwargs["parameters"]["temperature"] == 0.5
