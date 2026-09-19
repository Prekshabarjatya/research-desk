"""Worker: claims queued runs and advances their graph until it needs a human
or finishes. Stateless. Everything it needs to resume lives in the checkpointer,
so a crash or deploy loses at most the node that was executing."""

import logging
import signal
import threading
import time
from datetime import UTC, datetime

import httpx
from langgraph.types import Command

from app.citations import render_final
from app.config import settings
from app.models import Source
from app.store import RunStore

log = logging.getLogger("worker")


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 60}


def _pending_gate(snap):
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _drive(graph, payload, cfg, on_progress, should_stop=None) -> None:
    """Run the graph, reporting each finished node so the UI can show live progress.
    Stops between steps once should_stop() says the run was cancelled or deleted."""
    for chunk in graph.stream(payload, cfg, stream_mode="updates"):
        if on_progress is not None:
            for node, update in chunk.items():
                if node != "__interrupt__" and isinstance(update, dict):
                    on_progress(node, update)
        if should_stop is not None and should_stop():
            return


def advance(graph, run: dict, on_progress=None, should_stop=None) -> None:
    """Move the graph forward from wherever the checkpoint says it is.

    Decides from checkpointed state, not from the run's status, so it is correct
    after a crash at any point: fresh start, mid-run continue, or resume-from-gate."""
    cfg = _cfg(run["id"])
    snap = graph.get_state(cfg)
    if not snap.values and not snap.next:
        _drive(graph, {"prompt": run["prompt"]}, cfg, on_progress, should_stop)
    elif _pending_gate(snap) is not None:
        if run.get("resume") is None:
            return  # still waiting on the human; caller re-reads the gate
        _drive(graph, Command(resume=run["resume"]), cfg, on_progress, should_stop)
    elif snap.next:
        _drive(graph, None, cfg, on_progress, should_stop)  # crashed mid-run: continue from the last checkpoint


def summarize(values: dict) -> dict:
    crit = values.get("critique") or {}
    best = values.get("best")
    approved = bool(crit.get("approved"))
    draft, issues, note = values.get("draft", ""), crit.get("issues", []), ""
    if not approved and best:
        # The run ended on the revision cap. Return the last version that met every mechanical
        # requirement (length, sections, verified citations), not whatever the final rewrite produced.
        if best["draft"] != draft:
            note = (f"Returned draft v{best['revision']}, the last version that met all length, section and "
                    f"citation checks; later rewrites broke them.")
        draft, issues = best["draft"], best["issues"]
    sources = values.get("sources", [])
    srcs = [Source.model_validate(x) for x in sources]
    return {
        "topic": values.get("topic", ""),
        "thesis": values.get("thesis", ""),
        "constraints": values.get("constraints", {}),
        "draft": render_final(draft, srcs) if draft else "",
        "approved": approved,
        "needs_human_review": not approved,
        "open_issues": issues,
        "note": note,
        "revision_count": values.get("revision_count", 0),
        "tokens_used": values.get("tokens_used", 0),
        "verified_sources": len(sources),
        "log": values.get("log", []),
    }


def process(store: RunStore, graph, run: dict) -> None:
    stop = threading.Event()

    def beat():
        while not stop.wait(max(settings.stale_run_seconds / 4, 1)):
            store.heartbeat(run["id"])

    threading.Thread(target=beat, daemon=True).start()

    def gone_or_cancelled() -> bool:
        row = store.get(run["id"])
        return row is None or row["status"] == "cancelled"

    def on_progress(node: str, update: dict) -> None:
        entry = {"node": node, "log": update.get("log", []),
                 "at": datetime.now(UTC).isoformat(timespec="seconds")}
        for key in ("topic", "thesis"):  # lets the UI title the run before it finishes
            if update.get(key):
                entry[key] = update[key]
        store.add_progress(run["id"], entry)

    try:
        advance(graph, run, on_progress, gone_or_cancelled)
        if gone_or_cancelled():
            # Cancelled or deleted while working: record nothing. If it was deleted, the step that was
            # in flight may have written a checkpoint after the delete, so remove that too.
            if store.get(run["id"]) is None:
                try:
                    graph.checkpointer.delete_thread(run["id"])
                except Exception:
                    log.warning("run %s: could not clear leftover checkpoint", run["id"], exc_info=True)
            return
        snap = graph.get_state(_cfg(run["id"]))
        gate = _pending_gate(snap)
        if gate is not None:
            store.finish(run["id"], "awaiting_approval", gate=gate)
        else:
            store.finish(run["id"], "completed", result=summarize(snap.values))
    except Exception as exc:  # any failure becomes a recorded, retryable state
        log.exception("run %s failed", run["id"])
        partial = {}
        try:
            partial = summarize(graph.get_state(_cfg(run["id"])).values)
        except Exception:
            log.warning("run %s: could not read partial state", run["id"], exc_info=True)
        store.finish(run["id"], "failed", result=partial or None, error=f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()


def run_forever(store: RunStore, graph, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    while not stop.is_set():
        n = store.requeue_stale(settings.stale_run_seconds)
        if n:
            log.warning("re-queued %d stale run(s)", n)
        run = store.claim()
        if run is None:
            stop.wait(settings.worker_poll_seconds)
            continue
        log.info("claimed run %s (attempt %s)", run["id"], run["attempts"])
        process(store, graph, run)


def build_worker():
    """Connect to Postgres, prepare tables, and assemble the graph. Returns (store, graph)."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    from app.citations import verify_source
    from app.graph import build_graph
    from app.literature import search_all
    from app.llm import RoutedLLM, build_providers
    from app.nodes import Tools
    from app.store import PgRunStore

    pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    store = PgRunStore(pool)
    store.init()
    saver = PostgresSaver(pool)
    saver.setup()

    http = httpx.Client(headers={"User-Agent": "research-paper-agents/0.1"})
    def paced_search(query: str):
        time.sleep(0.5)  # free APIs throttle bursts; a short gap between queries avoids most 429s
        return search_all(query, http)

    tools = Tools(search=paced_search, verify=lambda s: verify_source(s, http))
    return store, build_graph(RoutedLLM(build_providers()), tools, saver)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store, graph = build_worker()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())  # finish the current run, then exit
    run_forever(store, graph, stop)


if __name__ == "__main__":
    main()
