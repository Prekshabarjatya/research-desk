import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.api import create_app
from app.config import settings
from app.graph import build_graph
from app.llm import LLMError
from app.store import MemoryRunStore
from app.worker import process
from tests.conftest import FakeLLM
from tests.test_graph import OUTLINE, approve, script, tools

PROMPT = "Write a 1500 word paper on AI routing in logistics."


def drain(store, graph):
    while (run := store.claim()) is not None:
        process(store, graph, run)


@pytest.fixture
def store():
    return MemoryRunStore()


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    return TestClient(create_app(store), headers={"Authorization": "Bearer secret"})


# ---- worker ----------------------------------------------------------------

def test_worker_drives_run_through_both_gates_to_completion(store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run = store.create(PROMPT)

    drain(store, graph)
    r = store.get(run["id"])
    assert r["status"] == "awaiting_approval" and r["gate"]["gate"] == "topic"

    assert store.approve(run["id"], approve())
    drain(store, graph)
    assert store.get(run["id"])["gate"]["gate"] == "thesis"

    assert store.approve(run["id"], approve())
    drain(store, graph)
    done = store.get(run["id"])
    assert done["status"] == "completed"
    assert done["result"]["approved"] and "## Introduction" in done["result"]["draft"]
    assert done["gate"] is None and done["resume"] is None


def test_failed_run_keeps_checkpoint_and_retry_resumes_without_redoing_work(store):
    llm = FakeLLM(script(planner=[LLMError("groq down"), OUTLINE]))
    graph = build_graph(llm, tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)

    failed = store.get(run["id"])
    assert failed["status"] == "failed" and "groq down" in failed["error"]

    assert store.retry(run["id"])
    drain(store, graph)
    assert store.get(run["id"])["status"] == "completed"
    assert llm.calls.count("analyst") == 1 and llm.calls.count("strategist") == 1
    assert llm.calls.count("planner") == 2  # only the failed step re-ran


def test_stale_running_run_is_requeued_and_continues_from_checkpoint(store):
    llm = FakeLLM(script())
    graph = build_graph(llm, tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)                        # stops at the topic gate
    store.approve(run["id"], approve())
    store.claim()                              # a worker claims it, then "dies": no finish, no heartbeat
    assert store.get(run["id"])["status"] == "running"

    assert store.requeue_stale(0) == 1
    drain(store, graph)
    assert store.get(run["id"])["gate"]["gate"] == "thesis"
    assert llm.calls.count("strategist") == 1


def test_budget_overrun_fails_the_run_with_partial_result(store, monkeypatch):
    from app import nodes
    monkeypatch.setattr(nodes.settings, "max_tokens_per_run", 150)
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)
    r = store.get(run["id"])
    assert r["status"] == "failed" and "BudgetExceeded" in r["error"]
    assert r["result"]["tokens_used"] == 200


def test_cancel_wins_over_a_worker_that_finishes_afterwards(store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run = store.create(PROMPT)
    claimed = store.claim()
    assert store.cancel(run["id"])
    process(store, graph, claimed)
    assert store.get(run["id"])["status"] == "cancelled"


# ---- API -------------------------------------------------------------------

def test_health_is_open_everything_else_needs_the_token(store, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    c = TestClient(create_app(store))
    assert c.get("/health").status_code == 200
    assert c.post("/runs", json={"prompt": PROMPT}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert c.post("/runs", json={"prompt": PROMPT}, headers=bad).status_code == 401


def test_unset_token_locks_the_api_instead_of_opening_it(store, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "")
    c = TestClient(create_app(store), headers={"Authorization": "Bearer "})
    assert c.post("/runs", json={"prompt": PROMPT}).status_code == 401


def test_create_validates_prompt_and_caps_active_runs(client, monkeypatch):
    assert client.post("/runs", json={"prompt": "short"}).status_code == 422
    monkeypatch.setattr(settings, "max_active_runs", 1)
    assert client.post("/runs", json={"prompt": PROMPT}).status_code == 202
    assert client.post("/runs", json={"prompt": PROMPT}).status_code == 429


def test_full_lifecycle_over_http(client, store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]

    assert client.get(f"/runs/{run_id}/draft").status_code == 404
    assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 409

    drain(store, graph)
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "awaiting_approval" and body["gate"]["gate"] == "topic"

    assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 202
    drain(store, graph)
    assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 202
    drain(store, graph)

    assert client.get(f"/runs/{run_id}").json()["status"] == "completed"
    draft = client.get(f"/runs/{run_id}/draft")
    assert draft.headers["content-type"].startswith("text/markdown") and "References" in draft.text


def test_unknown_run_and_bad_transitions(client):
    assert client.get("/runs/nope").status_code == 404
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]
    assert client.post(f"/runs/{run_id}/retry").status_code == 409
    assert client.post(f"/runs/{run_id}/cancel").status_code == 200
    assert client.post(f"/runs/{run_id}/cancel").status_code == 409


def test_capped_run_returns_the_best_draft_not_the_last_one(store, monkeypatch):
    from app.config import settings as cfg
    from app.worker import summarize
    from tests.test_graph import CONSTRAINTS, prose
    monkeypatch.setattr(cfg, "max_revisions", 3)
    long_ = "word " * 300
    llm = FakeLLM(script(
        analyst={**CONSTRAINTS, "max_words": 250},
        **{"writer:Introduction": [long_, prose("[S1]", 80)],
           "writer:Analysis": [long_, prose("[S2]", 80), long_]},   # v3 rewrites Analysis, overshoots
        critic={"approved": False, "issues": ["thin argument"], "sections_to_fix": ["Analysis"]},
    ))
    graph = build_graph(llm, tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)

    result = store.get(run["id"])["result"]
    assert result["revision_count"] == 3 and not result["approved"]
    assert "Returned draft v2" in result["note"]
    assert result["open_issues"] == ["thin argument"]           # the LLM critic's issues for v2
    assert "word word word" not in result["draft"]              # v3's overlong text is not returned
    assert summarize(graph.get_state({"configurable": {"thread_id": run["id"]}}).values)["draft"]


# ---- run list, live progress, UI hosting ------------------------------------

def test_worker_records_progress_per_node_while_running(store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    progress = store.get(run["id"])["progress"]
    assert [p["node"] for p in progress] == ["analyst", "propose_topic"]
    assert progress[1]["log"] and progress[1]["at"].endswith("+00:00")


def test_list_runs_is_compact_newest_first_and_authenticated(client, store):
    first = client.post("/runs", json={"prompt": PROMPT}).json()["id"]
    second = client.post("/runs", json={"prompt": PROMPT + " Second."}).json()["id"]
    rows = client.get("/runs").json()
    assert [r["id"] for r in rows] == [second, first]
    assert set(rows[0]) == {"id", "status", "prompt", "topic", "needs_human_review", "created_at", "updated_at"}
    assert TestClient(create_app(store)).get("/runs").status_code == 401


def test_completed_summary_carries_topic_thesis_and_constraints(client, store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]
    for _ in range(3):
        drain(store, graph)
        client.post(f"/runs/{run_id}/approve", json={"approved": True})
    body = client.get(f"/runs/{run_id}").json()
    assert body["result"]["topic"] == "AI routing" and "cuts cost" in body["result"]["thesis"]
    assert client.get("/runs").json()[0]["topic"] == "AI routing"


def test_ui_is_served_at_root_with_strict_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Research Desk" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/health").headers["content-security-policy"]  # API responses too


def test_memory_store_timestamps_match_postgres_format(store):
    run = store.create(PROMPT)
    assert str(run["created_at"]).endswith("+00:00")  # real datetimes, like Postgres, not raw epoch floats
    assert str(store.get(run["id"])["updated_at"]).endswith("+00:00")


def test_progress_carries_the_proposed_topic_so_the_ui_can_title_the_run(store):
    graph = build_graph(FakeLLM(script(strategist={"topic": "AI‑routing study", "search_queries": ["q"]})),
                        tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    with_topic = [p for p in store.get(run["id"])["progress"] if p.get("topic")]
    assert with_topic and with_topic[0]["topic"] == "AI-routing study"  # typography normalised too


# ---- hosted deploy pieces ---------------------------------------------------

def test_cors_allows_only_the_configured_ui_origin(store, monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", "https://desk.vercel.app/")
    c = TestClient(create_app(store))
    ok = c.options("/runs", headers={"Origin": "https://desk.vercel.app", "Access-Control-Request-Method": "POST",
                                      "Access-Control-Request-Headers": "authorization,content-type"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://desk.vercel.app"
    bad = c.options("/runs", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_no_cors_headers_by_default(store, monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", "")
    r = TestClient(create_app(store)).get("/health", headers={"Origin": "https://desk.vercel.app"})
    assert "access-control-allow-origin" not in r.headers


def test_ui_config_is_served_and_defaults_to_same_origin(client):
    r = client.get("/config.js")
    assert r.status_code == 200 and 'RD_API_BASE = ""' in r.text


def test_vercel_build_points_the_ui_at_the_api_and_allows_only_that_origin(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    root = Path(__file__).resolve().parent.parent
    work = tmp_path / "repo"
    shutil.copytree(root / "app" / "static", work / "app" / "static")
    (work / "scripts").mkdir()
    shutil.copy(root / "scripts" / "vercel_build.mjs", work / "scripts")
    run = lambda env: subprocess.run([node, "scripts/vercel_build.mjs"], cwd=work, env=env,
                                     capture_output=True, text=True, check=False)
    good = run({"API_BASE_URL": "https://desk-api.onrender.com/"})
    assert good.returncode == 0, good.stderr
    assert (work / "dist" / "config.js").read_text() == 'window.RD_API_BASE = "https://desk-api.onrender.com";\n'
    html = (work / "dist" / "index.html").read_text()
    assert "connect-src 'self' https://desk-api.onrender.com;" in html and "unsafe-inline" not in html
    assert run({"API_BASE_URL": "http://insecure.example"}).returncode == 1   # https only
    assert run({}).returncode == 1                                              # must be set


# ---- delete, and cancel that actually stops ----------------------------------

def _finished_run(client, store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]
    for _ in range(3):
        drain(store, graph)
        client.post(f"/runs/{run_id}/approve", json={"approved": True})
    return run_id


def test_delete_removes_a_finished_paper_and_needs_auth(client, store):
    run_id = _finished_run(client, store)
    assert client.get(f"/runs/{run_id}").json()["status"] == "completed"
    assert TestClient(create_app(store)).delete(f"/runs/{run_id}").status_code == 401
    assert client.delete(f"/runs/{run_id}").status_code == 204
    assert client.get(f"/runs/{run_id}").status_code == 404
    assert run_id not in [r["id"] for r in client.get("/runs").json()]
    assert client.delete(f"/runs/{run_id}").status_code == 404  # already gone


def test_an_active_run_must_be_cancelled_before_it_can_be_deleted(client):
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]        # queued
    assert client.delete(f"/runs/{run_id}").status_code == 409
    assert client.post(f"/runs/{run_id}/cancel").status_code == 200
    assert client.delete(f"/runs/{run_id}").status_code == 204


def test_a_run_waiting_on_you_can_be_deleted(client, store):
    graph = build_graph(FakeLLM(script()), tools(), MemorySaver())
    run_id = client.post("/runs", json={"prompt": PROMPT}).json()["id"]
    drain(store, graph)
    assert client.get(f"/runs/{run_id}").json()["status"] == "awaiting_approval"
    assert client.delete(f"/runs/{run_id}").status_code == 204


class CancelAfterFirstStep(MemoryRunStore):
    """Simulates a user pressing Cancel while the first step is finishing."""

    def add_progress(self, run_id, entry):
        super().add_progress(run_id, entry)
        if len(self._runs[run_id]["progress"]) == 1:
            self.cancel(run_id)


def test_cancel_stops_the_worker_before_it_spends_more_tokens():
    store = CancelAfterFirstStep()
    llm = FakeLLM(script())
    graph = build_graph(llm, tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    assert store.get(run["id"])["status"] == "cancelled"
    assert llm.calls == ["analyst"]  # the topic step never ran


class DeleteAfterFirstStep(MemoryRunStore):
    def add_progress(self, run_id, entry):
        super().add_progress(run_id, entry)
        if len(self._runs[run_id]["progress"]) == 1:
            self._runs[run_id]["status"] = "cancelled"
            self.delete(run_id)  # cancel then delete while the worker is mid-step


def test_deleting_mid_step_leaves_no_row_and_no_saved_progress():
    store = DeleteAfterFirstStep()
    llm = FakeLLM(script())
    graph = build_graph(llm, tools(), MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    assert store.get(run["id"]) is None
    assert llm.calls == ["analyst"]
    cfg = {"configurable": {"thread_id": run["id"]}}
    assert not graph.get_state(cfg).values  # leftover checkpoint was cleared


def test_cors_preflight_allows_delete_for_a_separately_hosted_ui(store, monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", "https://desk.vercel.app")
    r = TestClient(create_app(store)).options("/runs/x", headers={
        "Origin": "https://desk.vercel.app", "Access-Control-Request-Method": "DELETE",
        "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200 and "DELETE" in r.headers["access-control-allow-methods"]


def test_a_source_shortfall_fails_the_run_clearly_and_retry_resumes_at_the_scout(store):
    from app.models import Source
    from app.nodes import Tools
    live = {"up": False}
    pool = [Source(title=f"Paper {i}", authors=["A. B"], year=2022, doi=f"10.1/p{i}") for i in range(1, 4)]
    t = Tools(search=lambda q: list(pool) if live["up"] else [],
              verify=lambda s: s.model_copy(update={"verified": True}))
    llm = FakeLLM(script())
    graph = build_graph(llm, t, MemorySaver())
    run = store.create(PROMPT)
    drain(store, graph)
    store.approve(run["id"], approve())
    drain(store, graph)
    failed = store.get(run["id"])
    assert failed["status"] == "failed" and "SourceShortfall" in failed["error"]
    assert "thesis" not in llm.calls  # nothing was spent on drafting

    live["up"] = True                       # the search recovers
    assert store.retry(run["id"])
    drain(store, graph)
    assert store.get(run["id"])["gate"]["gate"] == "thesis"
    assert llm.calls.count("analyst") == 1 and llm.calls.count("strategist") == 1
