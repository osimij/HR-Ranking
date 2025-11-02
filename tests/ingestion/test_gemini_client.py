from types import SimpleNamespace

import pytest

from matcher.ingestion import gemini_client


class DummyChunk:
    def __init__(self, text: str):
        self.text = text


class DummyPart:
    @staticmethod
    def from_text(text: str):
        return text


class DummyContent:
    def __init__(self, role: str, parts):
        self.role = role
        self.parts = list(parts)


class DummyThinkingConfig:
    def __init__(self, thinking_budget: int):
        self.thinking_budget = thinking_budget


class DummyImageConfig:
    def __init__(self, image_size: str):
        self.image_size = image_size


class DummyGenerateContentConfig:
    def __init__(self, thinking_config, image_config):
        self.thinking_config = thinking_config
        self.image_config = image_config


class DummyTypes:
    Content = DummyContent
    Part = DummyPart
    ThinkingConfig = DummyThinkingConfig
    ImageConfig = DummyImageConfig
    GenerateContentConfig = DummyGenerateContentConfig


class DummyModels:
    def __init__(self, responses):
        self._responses = responses

    def generate_content_stream(self, **_kwargs):
        return iter(self._responses)


class DummyClient:
    def __init__(self, responses):
        self.models = DummyModels(responses)


@pytest.fixture(autouse=True)
def restore_module(monkeypatch):
    monkeypatch.setattr(gemini_client, "types", DummyTypes())
    monkeypatch.setattr(
        gemini_client,
        "genai",
        SimpleNamespace(Client=lambda api_key: DummyClient([DummyChunk("Hello"), DummyChunk(" World")])),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    yield
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_stream_text_completion_returns_joined_chunks():
    output = gemini_client.stream_text_completion("Hi there", model="test-model")
    assert output == "Hello World"


def test_build_client_uses_env_key():
    client = gemini_client.build_client()
    assert isinstance(client, DummyClient)
