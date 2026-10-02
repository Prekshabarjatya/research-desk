import time

import pytest

from app.llm import Completion, LLMError, RoutedLLM, _with_deadline, parse_json, prompt_schema
from app.models import Critique, Outline, TopicProposal


class Flaky:
    def __init__(self, name, replies):
        self.name, self.replies, self.n = name, list(replies), 0

    def invoke(self, system, user, tier):
        self.n += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Completion(r, 10)


def llm(*providers):
    return RoutedLLM(list(providers), retries=2, base_delay=0)


def test_retries_then_succeeds():
    p = Flaky("a", [RuntimeError("429"), "ok"])
    assert llm(p).complete("s", "u").text == "ok" and p.n == 2


def test_falls_back_to_second_provider_after_retries_exhausted():
    a, b = Flaky("a", [RuntimeError("x")] * 2), Flaky("b", ["from b"])
    assert llm(a, b).complete("s", "u").text == "from b"


def test_all_providers_failing_raises():
    with pytest.raises(LLMError):
        llm(Flaky("a", [RuntimeError("x")] * 2)).complete("s", "u")


def test_json_repair_attempt_then_success_sums_tokens():
    p = Flaky("a", ["not json", '```json\n{"approved": true}\n```'])
    crit, tokens = llm(p).complete_json("s", "u", Critique)
    assert crit.approved and tokens == 20


def test_json_invalid_twice_raises():
    with pytest.raises(LLMError):
        llm(Flaky("a", ["nope", "still nope"])).complete_json("s", "u", Critique)


class AuthError(RuntimeError):
    status_code = 401


class RateLimit(RuntimeError):
    status_code = 429


def test_auth_errors_are_not_retried_and_fall_through_immediately():
    a, b = Flaky("a", [AuthError("bad key")] * 3), Flaky("b", ["from b"])
    assert llm(a, b).complete("s", "u").text == "from b"
    assert a.n == 1  # one attempt, not `retries`


def test_rate_limits_are_still_retried():
    p = Flaky("a", [RateLimit("slow down"), "ok"])
    assert llm(p).complete("s", "u").text == "ok" and p.n == 2


class TooLarge(RuntimeError):
    status_code = 413


def test_oversized_requests_are_not_retried():
    a = Flaky("a", [TooLarge("too big")] * 3)
    with pytest.raises(LLMError):
        llm(a).complete("s", "u")
    assert a.n == 1


def test_deadline_stops_a_call_that_never_answers():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _with_deadline(lambda: time.sleep(5), 0.1)
    assert time.monotonic() - started < 1


def test_deadline_passes_results_and_errors_through():
    assert _with_deadline(lambda: "ok", 1) == "ok"
    with pytest.raises(TooLarge):
        _with_deadline(lambda: (_ for _ in ()).throw(TooLarge("too big")), 1)


def test_prompt_schema_drops_labels_but_keeps_a_field_named_title():
    schema = prompt_schema(Outline.model_json_schema())
    section = schema["$defs"]["SectionPlan"]
    assert "title" not in schema and "title" not in section
    assert section["properties"]["title"] == {"type": "string"}  # the real field survives
    assert "title" not in prompt_schema(TopicProposal.model_json_schema())["properties"]["topic"]


def test_a_string_field_wrapped_in_an_object_is_unwrapped():
    # Seen live: a small model answered {"topic": {"title": "..."}} for a string field.
    p = parse_json('{"topic": {"title": "Risk-Tiered Reporting"}, "search_queries": ["a"]}', TopicProposal)
    assert p.topic == "Risk-Tiered Reporting"
    o = parse_json('{"sections": [{"title": {"text": "Intro"}, "goal": "g"}]}', Outline)
    assert o.sections[0].title == "Intro"


def test_an_object_that_is_not_a_single_string_still_fails():
    with pytest.raises(ValueError):
        parse_json('{"topic": {"title": "A", "subtitle": "B"}}', TopicProposal)
