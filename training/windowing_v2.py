from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADER_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[A-ZÀ-Ỹ][^.!?]{0,80}:?\s*$)")


def tokenize_raw(text: str) -> list[tuple[str, int, int]]:
    """Tokenize trực tiếp RAW; tuyệt đối không NFC/NFD trước khi lấy offset."""
    return [(match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def nearest_header(text: str, char_start: int) -> str | None:
    prefix = text[:char_start]
    for line in reversed(prefix.splitlines()):
        if HEADER_RE.match(line):
            return line.strip()
    return None


def sliding_windows(
    text: str,
    max_words: int = 180,
    overlap_words: int = 45,
    include_header: bool = True,
    context_words: int = 32,
) -> list[dict]:
    if not (0 <= overlap_words < max_words):
        raise ValueError("overlap_words phải trong [0, max_words)")
    tokens = tokenize_raw(text)
    if not tokens:
        return []
    step = max_words - overlap_words
    windows = []
    for start in range(0, len(tokens), step):
        content = tokens[start:start + max_words]
        if not content:
            break
        header = nearest_header(text, content[0][1]) if include_header and start else None
        # Context format-agnostic: luôn mang một ít RAW ngay trước window. Header regex chỉ là
        # tín hiệu bổ sung khi header nằm xa, không phải điều kiện để inference hoạt động.
        prefix = tokens[max(0, start - max(0, context_words)):start] if start else []
        if header:
            header_tokens = tokenize_raw(header)
            header_words = [token for token, _, _ in header_tokens]
            prefix_words = [token for token, _, _ in prefix]
            width = len(header_words)
            header_already_visible = bool(width) and any(
                prefix_words[index:index + width] == header_words
                for index in range(max(0, len(prefix_words) - width + 1))
            )
            if not header_already_visible:
                prefix = header_tokens + prefix
        windows.append({
            "content": content,
            "prefix": prefix,
            "word_start": start,
            "char_range": [content[0][1], content[-1][2]],
        })
        if start + max_words >= len(tokens):
            break
    return windows


def entity_word_span(
    tokens: list[tuple[str, int, int]], start: int, end: int
) -> tuple[int, int] | None:
    indexes = [
        index for index, (_, token_start, token_end) in enumerate(tokens)
        if token_start >= start and token_end <= end
    ]
    if not indexes:
        return None
    first, last = indexes[0], indexes[-1]
    if tokens[first][1] != start or tokens[last][2] != end:
        return None
    return first, last + 1


def coverage_errors(text: str, entities: list[dict], **window_args) -> list[str]:
    windows = sliding_windows(text, **window_args)
    errors = []
    for index, entity in enumerate(entities):
        start, end = entity["position"]
        if not any(
            entity_word_span(window["content"], start, end) is not None
            for window in windows
        ):
            errors.append(f"entity[{index}]:not_covered")
    return errors
