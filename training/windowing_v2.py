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


def _subword_count(token: str, tokenizer, cache: dict[str, int]) -> int:
    if token not in cache:
        cache[token] = max(1, len(tokenizer.encode(token, add_special_tokens=False)))
    return cache[token]


def _fit_prefix(
    text: str,
    tokens: list[tuple[str, int, int]],
    start: int,
    *,
    include_header: bool,
    context_words: int,
    tokenizer,
    subword_cap: int,
    cache: dict[str, int],
) -> list[tuple[str, int, int]]:
    """Keep the nearest raw context and, when available, the section header.

    Header tokens are kept first.  Remaining budget is spent on the context closest to the
    content boundary.  Prefix offsets are never used for output entities, so header offsets may
    be local to the header string just like in the legacy implementation.
    """
    context = tokens[max(0, start - max(0, context_words)):start]
    header_tokens: list[tuple[str, int, int]] = []
    header = nearest_header(text, tokens[start][1]) if include_header and start else None
    if header:
        candidate = tokenize_raw(header)
        header_words = [token for token, _, _ in candidate]
        context_words_list = [token for token, _, _ in context]
        width = len(header_words)
        already_visible = bool(width) and any(
            context_words_list[index:index + width] == header_words
            for index in range(max(0, len(context_words_list) - width + 1))
        )
        if not already_visible:
            header_tokens = candidate

    def length(items: list[tuple[str, int, int]]) -> int:
        return sum(_subword_count(token, tokenizer, cache) for token, _, _ in items)

    # A pathological very long header must not consume the content itself.
    fitted_header: list[tuple[str, int, int]] = []
    used = 0
    for item in header_tokens:
        cost = _subword_count(item[0], tokenizer, cache)
        if used + cost > subword_cap:
            break
        fitted_header.append(item)
        used += cost
    remaining = subword_cap - used
    fitted_context: list[tuple[str, int, int]] = []
    for item in reversed(context):
        cost = _subword_count(item[0], tokenizer, cache)
        if cost > remaining:
            break
        fitted_context.append(item)
        remaining -= cost
    fitted_context.reverse()
    prefix = fitted_header + fitted_context
    assert length(prefix) <= subword_cap
    return prefix


def _boundary_aware_next_start(
    text: str,
    tokens: list[tuple[str, int, int]],
    current_start: int,
    content_end: int,
    overlap_words: int,
) -> int:
    """Prefer a sentence/line boundary in the overlap band.

    Tokenizer-budgeted windows have variable content length.  A fixed ``end-overlap`` start can
    cut a long result one token after its beginning.  Looking back by at most one additional
    overlap band preserves the same bounded-cost behaviour but greatly reduces split concepts
    without using gold entities.
    """
    nominal = max(current_start + 1, content_end - overlap_words)
    lower = max(current_start + 1, nominal - overlap_words)
    boundaries = []
    for index in range(lower, nominal + 1):
        if index <= 0 or index >= len(tokens):
            continue
        previous = tokens[index - 1]
        current = tokens[index]
        between = text[previous[2]:current[1]]
        if "\n" in between or previous[0] in {".", "!", "?", ";", ":"}:
            boundaries.append(index)
    return boundaries[0] if boundaries else nominal


def sliding_windows(
    text: str,
    max_words: int = 180,
    overlap_words: int = 45,
    include_header: bool = True,
    context_words: int = 32,
    tokenizer=None,
    max_len: int | None = None,
    prefix_subword_cap: int = 64,
    subword_cache: dict[str, int] | None = None,
) -> list[dict]:
    if not (0 <= overlap_words < max_words):
        raise ValueError("overlap_words phải trong [0, max_words)")
    tokens = tokenize_raw(text)
    if not tokens:
        return []
    if tokenizer is not None and (max_len is None or max_len < 4):
        raise ValueError("tokenizer-aware window cần max_len >= 4")
    step = max_words - overlap_words
    windows = []
    start = 0
    cache: dict[str, int] = subword_cache if subword_cache is not None else {}
    while start < len(tokens):
        if tokenizer is None:
            content = tokens[start:start + max_words]
            prefix = tokens[max(0, start - max(0, context_words)):start] if start else []
            header = nearest_header(text, content[0][1]) if include_header and start else None
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
            prefix_subwords = content_subwords = None
        else:
            token_budget = max_len - 2  # BOS + EOS
            first_cost = _subword_count(tokens[start][0], tokenizer, cache)
            prefix_cap = min(prefix_subword_cap, max(0, token_budget - first_cost))
            prefix = _fit_prefix(
                text, tokens, start,
                include_header=include_header,
                context_words=context_words,
                tokenizer=tokenizer,
                subword_cap=prefix_cap,
                cache=cache,
            ) if start else []
            prefix_subwords = sum(
                _subword_count(token, tokenizer, cache) for token, _, _ in prefix
            )
            available = token_budget - prefix_subwords
            content = []
            content_subwords = 0
            for item in tokens[start:start + max_words]:
                cost = _subword_count(item[0], tokenizer, cache)
                if content_subwords + cost > available:
                    break
                content.append(item)
                content_subwords += cost
        if not content:
            raise ValueError(
                f"Token RAW tại word={start} không vừa max_len={max_len}: {tokens[start][0]!r}"
            )
        end = start + len(content)
        windows.append({
            "content": content,
            "prefix": prefix,
            "word_start": start,
            "char_range": [content[0][1], content[-1][2]],
            "subword_length": (
                None if tokenizer is None else 2 + prefix_subwords + content_subwords
            ),
        })
        if end >= len(tokens):
            break
        start = (
            start + step if tokenizer is None
            else _boundary_aware_next_start(text, tokens, start, end, overlap_words)
        )
    return windows


def entity_centered_window(
    text: str,
    position: list[int] | tuple[int, int],
    tokenizer,
    *,
    max_len: int = 256,
    max_words: int = 180,
    left_context_words: int = 32,
    subword_cache: dict[str, int] | None = None,
) -> dict | None:
    """Build a gold-only rescue window when a normal boundary splits a long entity.

    This is a training supervision aid, not an inference boundary rule. The entity and context
    remain verbatim RAW with original offsets. If the entity itself exceeds the encoder budget,
    no rescue is possible and the caller keeps reporting the legacy coverage error.
    """
    tokens = tokenize_raw(text)
    span = entity_word_span(tokens, *position)
    if span is None:
        return None
    entity_start, entity_end = span
    cache = subword_cache if subword_cache is not None else {}
    costs = [_subword_count(token, tokenizer, cache) for token, _, _ in tokens]
    budget = max_len - 2
    used = sum(costs[entity_start:entity_end])
    if used > budget or entity_end - entity_start > max_words:
        return None
    start, end = entity_start, entity_end

    # Assertion/type cues are more often on the left; reserve up to 32 raw words there first.
    left_target = max(0, entity_start - left_context_words)
    while start > left_target and end - start < max_words and used + costs[start - 1] <= budget:
        start -= 1
        used += costs[start]
    # Fill the right side, then any remaining left budget.
    while end < len(tokens) and end - start < max_words and used + costs[end] <= budget:
        used += costs[end]
        end += 1
    while start > 0 and end - start < max_words and used + costs[start - 1] <= budget:
        start -= 1
        used += costs[start]
    content = tokens[start:end]
    return {
        "content": content,
        "prefix": [],
        "word_start": start,
        "char_range": [content[0][1], content[-1][2]],
        "subword_length": used + 2,
        "coverage_rescue": True,
    }


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
