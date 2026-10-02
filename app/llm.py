"""Provider-agnostic LLM wrapper: tier routing, retries with backoff, provider
fallback, and schema-validated JSON output with one repair attempt.

Nodes depend on the `LLM` interface only, so tests inject a fake and providers
can change without touching agent code."""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings

log = logging.getLogger("llm")
Tier = Literal["fast", "strong"]
WARN_PROMPT_TOKENS = 6000  # rough; keeps prompts small enough for free-tier providers
T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    """The run crossed its token ceiling. The run fails closed with partial state."""


# Client errors that retrying cannot fix (bad key, no credits, bad model name, malformed request).
# 413 is a too-large request: resending it unchanged cannot succeed.
# Rate limits (429) and server errors (5xx) are NOT here: those are worth retrying.
NON_RETRYABLE_STATUS = {400, 401, 402, 403, 404, 413, 422}


class LLMError(RuntimeError):
    """Every configured provider failed."""


@dataclass
class Completion:
    text: str
    tokens: int
    reasoning_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, user: str, *, tier: Tier = "fast", purpose: str = "") -> Completion: ...

    def complete_json(
        self, system: str, user: str, schema: type[T], *, tier: Tier = "fast", purpose: str = ""
    ) -> tuple[T, int]: ...


class Provider(Protocol):
    name: str

    def invoke(self, system: str, user: str, tier: Tier) -> Completion: ...


def _with_deadline(fn, seconds: float):
    """Run fn() but give up after `seconds` of wall-clock time. The HTTP client's timeout only
    limits the gap between bytes, and OpenRouter can keep a slow request alive indefinitely.
    The abandoned call finishes (or fails) in its daemon thread and its result is dropped."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001  re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no answer within {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class _LangChainProvider:
    def __init__(self, name: str, chat_fast, chat_strong):
        self.name = name
        self._chat = {"fast": chat_fast, "strong": chat_strong}

    def invoke(self, system: str, user: str, tier: Tier) -> Completion:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = _with_deadline(lambda: self._chat[tier].invoke(messages), settings.llm_timeout_seconds)
        content = resp.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        usage = getattr(resp, "usage_metadata", None) or {}
        return Completion(
            text=content.strip(),
            tokens=int(usage.get("total_tokens", 0)),
            reasoning_tokens=int((usage.get("output_token_details") or {}).get("reasoning", 0)),
        )


def _openai_compatible(name: str, base_url: str, api_key: str, fast: str, strong: str,
                       extra_body: dict | None = None) -> Provider:
    from langchain_openai import ChatOpenAI

    def mk(model: str, temp: float):
        return ChatOpenAI(
            model=model, base_url=base_url, api_key=api_key, temperature=temp, extra_body=extra_body,
            timeout=settings.llm_timeout_seconds,  # idle limit; _with_deadline caps the whole call
            max_retries=0,  # RoutedLLM retries and logs each try; client retries would multiply them unseen
        )

    return _LangChainProvider(name, mk(fast or strong, 0.2), mk(strong or fast, 0.3))


def build_providers() -> list[Provider]:
    providers: list[Provider] = []
    if settings.openrouter_api_key:
        # Every call is a fresh system + user exchange, so there is no reasoning_details to pass back.
        body: dict = {"reasoning": {"enabled": settings.openrouter_reasoning}}
        if settings.openrouter_provider:
            body["provider"] = {"only": [settings.openrouter_provider], "allow_fallbacks": False}
        providers.append(
            _openai_compatible(
                "openrouter", settings.openrouter_base_url, settings.openrouter_api_key,
                settings.openrouter_model_fast, settings.openrouter_model_strong, body,
            )
        )
    if settings.fallback_base_url and settings.fallback_api_key:
        providers.append(
            _openai_compatible(
                "fallback", settings.fallback_base_url, settings.fallback_api_key,
                settings.fallback_model_fast, settings.fallback_model_strong,
            )
        )
    return providers


class RoutedLLM:
    def __init__(self, providers: list[Provider], *, retries: int = 3, base_delay: float = 2.0):
        if not providers:
            raise LLMError("No LLM provider configured: set OPENROUTER_API_KEY (and optionally FALLBACK_*).")
        self.providers = providers
        self.retries = retries
        self.base_delay = base_delay

    def complete(self, system: str, user: str, *, tier: Tier = "fast", purpose: str = "") -> Completion:
        estimate = (len(system) + len(user)) // 3  # ~3 chars/token is a safe over-estimate for English
        if estimate > WARN_PROMPT_TOKENS:
            log.warning("large prompt for '%s': ~%d tokens; provider limits may reject it", purpose, estimate)
        errors: list[str] = []
        for provider in self.providers:
            for attempt in range(self.retries):
                started = time.monotonic()
                try:
                    result = provider.invoke(system, user, tier)
                except Exception as exc:  # noqa: BLE001  rate limit, timeout, 5xx; not distinguishable across SDKs
                    errors.append(f"{provider.name}: {exc}")
                    log.warning(
                        "%s %s '%s' failed after %.0fs (try %d/%d): %s", provider.name, tier, purpose,
                        time.monotonic() - started, attempt + 1, self.retries, exc,
                    )
                    if getattr(exc, "status_code", None) in NON_RETRYABLE_STATUS:
                        break  # go straight to the next provider
                    time.sleep(self.base_delay * (2**attempt))
                    continue
                log.info(
                    "%s %s '%s': %.0fs, %d tokens (%d reasoning)", provider.name, tier, purpose,
                    time.monotonic() - started, result.tokens, result.reasoning_tokens,
                )
                return result
            # exhausted retries on this provider, fall through to the next one
        raise LLMError(f"All providers failed for '{purpose}': {errors[-3:]}")

    def complete_json(
        self, system: str, user: str, schema: type[T], *, tier: Tier = "fast", purpose: str = ""
    ) -> tuple[T, int]:
        instruction = (
            f"{system}\n\nRespond with ONLY a JSON object matching this schema, no prose, no code fences:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        first = self.complete(instruction, user, tier=tier, purpose=purpose)
        try:
            return parse_json(first.text, schema), first.tokens
        except (ValueError, ValidationError) as exc:
            repair = self.complete(
                instruction,
                f"{user}\n\nYour previous reply was invalid ({exc}). Previous reply:\n{first.text}\n"
                "Return corrected JSON only.",
                tier=tier,
                purpose=f"{purpose}:repair",
            )
            try:
                return parse_json(repair.text, schema), first.tokens + repair.tokens
            except (ValueError, ValidationError) as exc2:
                raise LLMError(f"'{purpose}' returned invalid JSON twice: {exc2}") from exc2


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str, schema: type[T]) -> T:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return schema.model_validate(json.loads(cleaned[start : end + 1]))
