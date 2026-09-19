"""Literature search clients. OpenAlex is primary (free), Semantic Scholar is best-effort, and
Crossref search is the fallback when the others come back thin. All return unverified Sources;
the citation verifier decides what may be cited.

Shared cloud IPs get rate limited (HTTP 429) by these free APIs, so every call retries with backoff
and a failure is logged, never silent."""

import logging
import re
import time

import httpx

from app.config import settings
from app.models import Source

log = logging.getLogger("literature")
_sleep = time.sleep  # patched in tests
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_WAIT = 15.0


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return min(float(resp.headers["Retry-After"]), MAX_WAIT)
    except (KeyError, ValueError):
        return None


def _get(client: httpx.Client, url: str, *, params=None, headers=None, retries: int = 3) -> httpx.Response:
    """GET, retrying rate limits and server errors with backoff (honouring Retry-After)."""
    host = httpx.URL(url).host
    for attempt in range(retries + 1):
        resp = client.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code in RETRY_STATUSES and attempt < retries:
            wait = _retry_after(resp) or min(2.0**attempt, 8.0)
            log.warning("%s answered %s; retrying in %.0fs (%d/%d)", host, resp.status_code, wait, attempt + 1, retries)
            _sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise AssertionError("unreachable")


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, client: httpx.Client, limit: int = 10) -> list[Source]:
    params = {
        "search": query,
        "per-page": limit,
        "filter": "type:article,has_doi:true",
        "select": "title,doi,publication_year,authorships,abstract_inverted_index,primary_location",
    }
    if settings.openalex_mailto:
        params["mailto"] = settings.openalex_mailto
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    resp = _get(client, "https://api.openalex.org/works", params=params)
    out = []
    for w in resp.json().get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append(
            Source(
                title=w.get("title") or "",
                authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
                year=w.get("publication_year"),
                doi=w.get("doi") or "",
                abstract=_openalex_abstract(w.get("abstract_inverted_index")),
                venue=loc.get("display_name") or "",
                url=w.get("doi") or "",
            )
        )
    return [s for s in out if s.title]


def search_semantic_scholar(query: str, client: httpx.Client, limit: int = 10) -> list[Source]:
    headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else {}
    # Best-effort: without a key it is heavily rate limited, so do not spend time retrying it.
    resp = _get(
        client, "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": limit, "fields": "title,authors,year,abstract,venue,externalIds,url"},
        headers=headers, retries=3 if settings.semantic_scholar_api_key else 0,
    )
    out = []
    for p in resp.json().get("data", []):
        out.append(
            Source(
                title=p.get("title") or "",
                authors=[a["name"] for a in p.get("authors", [])],
                year=p.get("year"),
                doi=(p.get("externalIds") or {}).get("DOI", ""),
                abstract=p.get("abstract") or "",
                venue=p.get("venue") or "",
                url=p.get("url") or "",
            )
        )
    return [s for s in out if s.title]


def _plain(jats: str) -> str:
    text = re.sub(r"<[^>]+>", " ", jats or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)  # tag removal leaves a space before punctuation
    return re.sub(r"^Abstract\s*", "", text)


def search_crossref(query: str, client: httpx.Client, limit: int = 10) -> list[Source]:
    """Fallback. Crossref holds the DOI records themselves, so what it returns is real by construction.
    Restricted to journal articles that have an abstract, since the writer relies on abstracts."""
    params = {
        "query.bibliographic": query,
        "rows": limit,
        "filter": "type:journal-article,has-abstract:true",
        "select": "DOI,title,author,issued,container-title,abstract,URL",
    }
    if settings.crossref_mailto:
        params["mailto"] = settings.crossref_mailto
    resp = _get(client, "https://api.crossref.org/works", params=params)
    out = []
    for w in resp.json().get("message", {}).get("items", []):
        year = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        out.append(
            Source(
                title=" ".join(w.get("title") or []),
                authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in w.get("author", []) if a.get("family")],
                year=year,
                doi=w.get("DOI", ""),
                abstract=_plain(w.get("abstract", "")),
                venue=" ".join(w.get("container-title") or []),
                url=w.get("URL", ""),
            )
        )
    return [s for s in out if s.title and s.doi]


def search_all(query: str, client: httpx.Client, limit: int = 10) -> list[Source]:
    """One failing source must not fail the search, but it must be visible in the logs."""
    results: list[Source] = []
    for fn in (search_openalex, search_semantic_scholar):
        try:
            results.extend(fn(query, client, limit))
        except httpx.HTTPError as exc:
            log.warning("%s failed for %r: %s", fn.__name__, query, exc)
    if len(results) < 3:
        try:
            results.extend(search_crossref(query, client, limit))
        except httpx.HTTPError as exc:
            log.warning("search_crossref failed for %r: %s", query, exc)
    return results
