import pytest

from matcher.normalization.text import RussianTextNormalizer, normalize_lines


@pytest.fixture(scope="module")
def normalizer() -> RussianTextNormalizer:
    return RussianTextNormalizer()


def test_normalize_tokens_lemmatizes_and_filters_digits(normalizer: RussianTextNormalizer) -> None:
    tokens = normalizer.normalize_tokens("Разработчик Python 2024 года, работал с данными.")
    assert "разработчик" in tokens
    assert "python" in tokens
    assert all(token != "2024" for token in tokens)


def test_normalize_text_handles_mixed_alphabet(normalizer: RussianTextNormalizer) -> None:
    text = "Mocква"  # mix of latin and cyrillic characters
    normalized = normalizer.normalize_text(text)
    assert normalized == "mocква"


def test_normalize_lines_skips_empty_entries(normalizer: RussianTextNormalizer) -> None:
    lines = [" Python разработчик ", "", " Аналитик данных "]
    normalized = normalize_lines(lines, normalizer)
    assert normalized[0].startswith("python")
    assert len(normalized) == 2
