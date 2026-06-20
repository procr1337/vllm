# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for content duplication at the engine-based
reasoning->content transition in :class:`DelegatingParser`.

Background
----------
When a server enables an engine-based reasoning parser *and* an engine-based
tool parser (e.g. ``--reasoning-parser glm45 --tool-call-parser glm47``), a
no-tools request streams its post-reasoning text through the shared delegating
parser path.

The serving layer streams via an *incremental* detokenizer, which holds tokens
back across steps (so multi-token unicode / leading spaces render correctly).
That means the incremental ``delta_text`` lags the ``delta_token_ids``.

A regression once existed where the transition reconstructed content with a
*stateless* ``model_tokenizer.decode(current_token_ids)`` instead of the
already-streamed text. Under hold-back, that re-decoded content that the
incremental stream had not yet emitted, and the stream then emitted it again ->
duplicated content (the production symptom was `" limit < limit "`).

These tests model the hold-back and assert the *accumulated-content invariant*:
buffering across deltas is fine, but the concatenation of all streamed content
must equal the model's content exactly (no duplication, no loss), independent
of how the incremental detokenizer lags.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.replay_harness import collect_output, make_mock_tokenizer
from tests.parser.engine.trace_builder import Scenario, _build_glm47_moe
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import (
    Glm47MoeParserReasoningAdapter,
    Glm47MoeParserToolAdapter,
)


class _Glm47Delegating(DelegatingParser):
    """Mirrors ``--reasoning-parser glm45 --tool-call-parser glm47``: two
    engine-based adapters, so ``_engine_based`` is True."""

    reasoning_parser_cls = Glm47MoeParserReasoningAdapter
    tool_parser_cls = Glm47MoeParserToolAdapter


def _no_tools_request() -> ChatCompletionRequest:
    # Production-shaped request: no ``tools`` field, no ``tool_choice``.
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = None
    req.tool_choice = None
    return req


def _stream_with_holdback(
    parser: DelegatingParser,
    tokens: list[tuple[int, str]],
    request: ChatCompletionRequest,
    holdback: int,
) -> str:
    """Replay ``tokens`` one-per-step, but release each token's *text* only
    after ``holdback`` further tokens have been sampled -- modelling vLLM's
    incremental detokenizer read_offset lag (made easier to hit by MTP /
    speculative decoding). ``delta_token_ids`` always carries the just-sampled
    token id. Returns the accumulated streamed content."""
    pending: deque[str] = deque()
    deltas = []
    n = len(tokens)
    for k, (tid, text) in enumerate(tokens):
        pending.append(text)
        release = ""
        while len(pending) > holdback:
            release += pending.popleft()
        is_last = k == n - 1
        if is_last:
            while pending:
                release += pending.popleft()
        deltas.append(parser.parse_delta(release, [tid], request, finished=is_last))
    return collect_output(deltas).content


# Content shapes that exercise the '<' lexer-buffering path (a '<' could begin
# '<tool_call>') as well as plain text and whitespace-heavy code.
_CONTENTS = [
    " if limit < 2:\n        return []",
    " limit < 2",
    " a < b < c < d",
    " plain content without angle brackets",
    " trailing angle <",
    " 1 < 2 and 3 < 4\n   done",
]

_REASONINGS = ["let me think about it", "r", ""]

_HOLDBACKS = [0, 1, 2, 3]


@pytest.mark.parametrize("content", _CONTENTS, ids=lambda c: repr(c))
@pytest.mark.parametrize("reasoning", _REASONINGS, ids=lambda r: f"reason={r!r}")
@pytest.mark.parametrize("holdback", _HOLDBACKS, ids=lambda h: f"holdback={h}")
def test_no_tools_post_reasoning_content_not_duplicated(content, reasoning, holdback):
    """Post-reasoning no-tools content must stream exactly once, regardless of
    how far the incremental detokenizer lags the token ids."""
    sample = _build_glm47_moe(
        Scenario(id="dup", description="", reasoning=reasoning, content=content),
        validate=False,
    )
    parser = _Glm47Delegating(make_mock_tokenizer(sample), None)

    streamed = _stream_with_holdback(
        parser, sample.tokens, _no_tools_request(), holdback
    )

    assert streamed == sample.expected_content, (
        "Accumulated streamed content diverged from the model output "
        f"(holdback={holdback}).\n"
        f"  expected: {sample.expected_content!r}\n"
        f"  streamed: {streamed!r}"
    )


def test_holdback_matches_no_holdback():
    """The streamed content must be invariant to incremental-detokenizer lag:
    every hold-back depth must produce the same content as no hold-back."""
    content = " if limit < 2:\n        return []"
    sample = _build_glm47_moe(
        Scenario(id="dup", description="", reasoning="let me think", content=content),
        validate=False,
    )

    baseline = _stream_with_holdback(
        _Glm47Delegating(make_mock_tokenizer(sample), None),
        sample.tokens,
        _no_tools_request(),
        holdback=0,
    )
    assert baseline == sample.expected_content

    for holdback in (1, 2, 3):
        got = _stream_with_holdback(
            _Glm47Delegating(make_mock_tokenizer(sample), None),
            sample.tokens,
            _no_tools_request(),
            holdback,
        )
        assert got == baseline, (
            f"hold-back={holdback} changed the streamed content "
            f"(duplication/loss):\n  baseline: {baseline!r}\n  got:      {got!r}"
        )
