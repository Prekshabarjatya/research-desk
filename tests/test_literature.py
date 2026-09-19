import logging

import httpx
import pytest

from app import literature
from app.literature import search_all, search_crossref, search_openalex, search_semantic_scholar

OA = {"results": [{"title": "Lost in the Middle", "doi": "https://doi.org/10.1/a", "publication_year": 2024,
                   "authorships": [{"author": {"display_name": "N. Liu"}}],
                   "abstract_inverted_index": {"Models": [0], "struggle": [1]}, "primary_location": {}}]}
CR = {"message": {"items": [{
    "DOI": "10.2/b", "title": ["A Crossref Paper"], "author": [{"given": "Ann", "family": "Ames"}],
    "issued": {"date-parts": [[2023, 5]]}, "container-title": ["Journal X"], "URL": "https://doi.org/10.2/b",
    "abstract": "<jats:title>Abstract</jats:title><jats:p>Context   position <jats:italic>matters</jats:italic>.</jats:p>"}]}}


class Server:
    """Scripted HTTP server: per host, a list of responses consumed in order (the last one repeats)."""

    def __init__(self, **by_host):
        self.by_host = {k: list(v) for k, v in by_host.items()}
        self.calls: list[str] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        host = request.url.host
        self.calls.append(host)
        queue = self.by_host[host]
        status, body, headers = (queue.pop(0) if len(queue) > 1 else queue[0])
        return httpx.Response(status, json=body, headers=headers)

    def count(self, host):
        return self.calls.count(host)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(literature, "_sleep", waits.append)
    return waits


def test_rate_limit_is_retried_then_succeeds(no_sleep):
    srv = Server(**{"api.openalex.org": [(429, {}, {}), (429, {}, {}), (200, OA, {})]})
    out = search_openalex("q", srv.client)
    assert [s.title for s in out] == ["Lost in the Middle"] and out[0].abstract == "Models struggle"
    assert srv.count("api.openalex.org") == 3 and len(no_sleep) == 2


def test_persistent_rate_limit_gives_up_after_bounded_retries(no_sleep):
    srv = Server(**{"api.openalex.org": [(429, {}, {})]})
    with pytest.raises(httpx.HTTPStatusError):
        search_openalex("q", srv.client)
    assert srv.count("api.openalex.org") == 4  # first try plus three retries, not an endless loop


def test_retry_after_is_honoured_but_capped(no_sleep):
    srv = Server(**{"api.openalex.org": [(429, {}, {"Retry-After": "3"}), (429, {}, {"Retry-After": "900"}),
                                          (200, OA, {})]})
    search_openalex("q", srv.client)
    assert no_sleep == [3.0, 15.0]


def test_client_errors_are_not_retried(no_sleep):
    srv = Server(**{"api.openalex.org": [(400, {}, {})]})
    with pytest.raises(httpx.HTTPStatusError):
        search_openalex("q", srv.client)
    assert srv.count("api.openalex.org") == 1 and no_sleep == []


def test_semantic_scholar_is_not_retried_without_a_key(no_sleep, monkeypatch):
    monkeypatch.setattr(literature.settings, "semantic_scholar_api_key", "")
    srv = Server(**{"api.semanticscholar.org": [(429, {}, {})]})
    with pytest.raises(httpx.HTTPStatusError):
        search_semantic_scholar("q", srv.client)
    assert srv.count("api.semanticscholar.org") == 1


def test_crossref_results_are_parsed_and_jats_markup_stripped():
    srv = Server(**{"api.crossref.org": [(200, CR, {})]})
    (s,) = search_crossref("q", srv.client)
    assert (s.title, s.authors, s.year, s.doi, s.venue) == ("A Crossref Paper", ["Ann Ames"], 2023, "10.2/b", "Journal X")
    assert s.abstract == "Context position matters."


def test_search_all_falls_back_to_crossref_when_the_primary_sources_are_blocked(no_sleep):
    srv = Server(**{"api.openalex.org": [(429, {}, {})], "api.semanticscholar.org": [(429, {}, {})],
                    "api.crossref.org": [(200, CR, {})]})
    out = search_all("q", srv.client)
    assert [s.doi for s in out] == ["10.2/b"]


def test_search_all_skips_the_fallback_when_the_primary_source_is_healthy():
    many = {"results": OA["results"] * 3}
    srv = Server(**{"api.openalex.org": [(200, many, {})], "api.semanticscholar.org": [(200, {"data": []}, {})]})
    assert len(search_all("q", srv.client)) == 3 and srv.count("api.crossref.org") == 0


def test_search_failures_are_logged_not_silent(caplog, no_sleep):
    srv = Server(**{"api.openalex.org": [(429, {}, {})], "api.semanticscholar.org": [(429, {}, {})],
                    "api.crossref.org": [(429, {}, {})]})
    with caplog.at_level(logging.WARNING, logger="literature"):
        assert search_all("some query", srv.client) == []
    text = caplog.text
    assert "search_openalex failed" in text and "search_crossref failed" in text and "429" in text
