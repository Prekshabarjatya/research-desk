import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph
from app.llm import BudgetExceeded
from app.models import Source
from app.nodes import Tools
from tests.conftest import FakeLLM

CONSTRAINTS = {"min_words": 0, "max_words": 0, "citation_style": "APA", "required_sections": [], "goals": []}
OUTLINE = {"sections": [
    {"title": "Introduction", "goal": "frame", "source_ids": ["S1"], "target_words": 50},
    {"title": "Analysis", "goal": "argue", "source_ids": ["S2", "S3"], "target_words": 50},
]}


def prose(tail: str = "", n: int = 45) -> str:
    """A realistic-length section body; sections under 40 words are rejected as nearly empty."""
    return "lorem " * n + tail


def script(**over):
    base = {
        "analyst": CONSTRAINTS,
        "strategist": {"topic": "AI routing", "rationale": "r", "search_queries": ["routing"]},
        "thesis": "AI routing cuts cost.",
        "planner": OUTLINE,
        "writer:Introduction": prose("[S1]"),
        "writer:Analysis": prose("[S2] and [S3]"),
        "critic": {"approved": True},
    }
    base.update(over)
    return base


def tools(n=3):
    pool = [Source(title=f"Paper {i}", authors=["A. B"], year=2022, doi=f"10.1/p{i}") for i in range(1, n + 1)]
    return Tools(search=lambda q: list(pool), verify=lambda s: s.model_copy(update={"verified": True}))


def run(llm, t=None, decisions=()):
    graph = build_graph(llm, t or tools(), MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 60}
    out = graph.invoke({"prompt": "Write about AI routing"}, cfg)
    for d in decisions:
        assert "__interrupt__" in out, "expected a human gate"
        out = graph.invoke(Command(resume=d), cfg)
    return graph, cfg, out


def approve(**kw):
    return {"approved": True, **kw}


def test_happy_path_pauses_at_both_gates_then_completes():
    llm = FakeLLM(script())
    graph = build_graph(llm, tools(), MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 60}

    out = graph.invoke({"prompt": "Write about AI routing"}, cfg)
    assert out["__interrupt__"][0].value["gate"] == "topic"
    out = graph.invoke(Command(resume=approve()), cfg)
    assert out["__interrupt__"][0].value["gate"] == "thesis"
    out = graph.invoke(Command(resume=approve()), cfg)

    assert "__interrupt__" not in out
    assert "## Introduction" in out["draft"] and "## References" in out["draft"]
    assert out["tokens_used"] > 0 and len(out["revisions"]) == 1


def test_gate_resume_does_not_rerun_llm_calls():
    llm = FakeLLM(script())
    run(llm, decisions=[approve(), approve()])
    assert llm.calls.count("strategist") == 1 and llm.calls.count("thesis") == 1


def test_rejected_topic_loops_back_with_feedback():
    llm = FakeLLM(script())
    run(llm, decisions=[{"approved": False, "feedback": "too broad"}, approve(), approve()])
    assert llm.calls.count("strategist") == 2


def test_too_few_verified_sources_stops_the_run_before_any_drafting_tokens_are_spent():
    from app.nodes import SourceShortfall
    t = Tools(search=lambda q: [Source(title="Fake", doi="10.9/fake")],
              verify=lambda s: s.model_copy(update={"verified": False}))
    llm = FakeLLM(script())
    graph = build_graph(llm, t, MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 60}
    graph.invoke({"prompt": "x"}, cfg)
    with pytest.raises(SourceShortfall, match="Only 0 verifiable sources"):
        graph.invoke(Command(resume=approve()), cfg)
    assert not {"thesis", "planner"} & set(llm.calls) and not any(c.startswith("writer") for c in llm.calls)


def test_hallucinated_citation_is_caught_without_an_llm_critic_call():
    llm = FakeLLM(script(**{"writer:Introduction": prose("[S99]")}, critic={"approved": True}))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert llm.calls.count("critic") == 0  # deterministic check failed first
    assert out["revision_count"] == 4       # hit the cap, still failing
    assert "S99" in " ".join(out["critique"]["issues"])


def test_revision_cap_stops_a_critic_that_never_approves():
    llm = FakeLLM(script(critic={"approved": False, "issues": ["meh"], "sections_to_fix": ["Analysis"]}))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert out["revision_count"] == 4 and len(out["revisions"]) == 4


def test_revision_only_rewrites_flagged_sections():
    llm = FakeLLM(script(critic=[
        {"approved": False, "issues": ["x"], "sections_to_fix": ["Analysis"]},
        {"approved": True},
    ]))
    run(llm, decisions=[approve(), approve()])
    assert llm.calls.count("writer:Introduction") == 1 and llm.calls.count("writer:Analysis") == 2


def test_needs_more_sources_routes_through_scout_and_replans():
    llm = FakeLLM(script(critic=[
        {"approved": False, "issues": ["thin"], "needs_more_sources": True, "search_hint": "cost data"},
        {"approved": True},
    ]))
    run(llm, decisions=[approve(), approve()])
    assert llm.calls.count("planner") == 2 and llm.calls.count("thesis") == 1


def test_budget_exceeded_fails_closed(monkeypatch):
    from app import nodes
    monkeypatch.setattr(nodes.settings, "max_tokens_per_run", 150)
    llm = FakeLLM(script())
    graph = build_graph(llm, tools(), MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 60}
    graph.invoke({"prompt": "x"}, cfg)  # analyst + strategist use 200 tokens, over the 150 limit
    with pytest.raises(BudgetExceeded):
        graph.invoke(Command(resume=approve()), cfg)  # the next node refuses to spend more


def test_word_limits_count_the_body_not_the_reference_list():
    from app.checks import body_text, hard_issues, word_count
    from app.models import Constraints
    draft = "word " * 100 + "\n## References\n\n" + "ref " * 500
    assert word_count(body_text(draft)) == 100
    assert not hard_issues(draft, {}, Constraints(max_words=150), [])
    assert hard_issues(draft, {}, Constraints(max_words=50), [])


def test_planner_targets_are_rescaled_to_the_paper_length():
    from app.checks import scale_outline
    from app.models import Constraints, SectionPlan
    plan = [SectionPlan(title=t, goal="g", target_words=w) for t, w in [("A", 900), ("B", 900), ("C", 600)]]
    out = scale_outline(plan, Constraints(min_words=1200, max_words=1500))  # target 1350, plan sums to 2400
    assert abs(sum(s.target_words for s in out) - 1350) <= 3
    assert [s.target_words for s in scale_outline(plan, Constraints())] == [900, 900, 600]  # no limits: untouched


def test_writer_shrinks_sections_after_an_overlong_draft():
    long_a, long_b = "word " * 400, "word " * 400
    short_a, short_b = prose("[S1]", 100), prose("[S2]", 100)
    llm = FakeLLM(script(
        constraints=None,
        analyst={**CONSTRAINTS, "max_words": 500},
        planner=OUTLINE,
        **{"writer:Introduction": [long_a, short_a], "writer:Analysis": [long_b, short_b]},
        critic={"approved": True},
    ))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert out["revision_count"] == 2
    from app.checks import body_text, word_count
    assert word_count(body_text(out["draft"])) <= 500


def test_scout_keeps_only_the_most_relevant_sources_up_to_the_cap(monkeypatch):
    from app import nodes
    monkeypatch.setattr(nodes.settings, "max_sources", 2)
    monkeypatch.setattr(nodes.settings, "min_verified_sources", 1)
    pool = [
        Source(title="Edge intelligence for phones", doi="10.1/a", abstract="edge computing"),
        Source(title="AI routing cuts logistics cost", doi="10.1/b", abstract="routing optimization logistics"),
        Source(title="Routing optimization in logistics", doi="10.1/c", abstract="AI routing"),
        Source(title="Cooking with cast iron", doi="10.1/d", abstract="recipes"),
    ]
    verified = []
    t = Tools(search=lambda q: list(pool),
              verify=lambda s: (verified.append(s.doi), s.model_copy(update={"verified": True}))[1])
    _, _, out = run(FakeLLM(script()), t, decisions=[approve()])
    titles = {s["title"] for s in out["sources"]}
    assert titles == {"AI routing cuts logistics cost", "Routing optimization in logistics"}
    assert len(verified) == 2  # stopped verifying once full
    assert [s["id"] for s in out["sources"]] == ["S1", "S2"]


def test_empty_section_output_is_retried_before_it_reaches_the_draft():
    good = "A solid paragraph of real content. " * 20 + "[S2]"
    llm = FakeLLM(script(**{"writer:Analysis": ["ok", good]}))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert llm.calls.count("writer:Analysis") == 2
    assert "solid paragraph" in out["sections"]["Analysis"]


def test_a_persistently_empty_section_is_a_hard_issue_that_targets_only_that_section():
    from app.checks import hard_findings
    from app.models import Constraints
    sections = {"Introduction": "word " * 100, "Analysis": "ok"}
    findings = hard_findings("\n\n".join(f"## {t}\n\n{x}" for t, x in sections.items()), sections,
                             Constraints(), [])
    assert [(i.split("'")[1], secs) for i, secs in findings] == [("Analysis", ["Analysis"])]


def test_citation_issue_targets_only_the_section_containing_the_bad_marker():
    from app.checks import hard_findings
    from app.models import Constraints
    sections = {"Introduction": "fine " * 60, "Analysis": "bad claim [S99] " + "word " * 60}
    draft = "\n\n".join(f"## {t}\n\n{x}" for t, x in sections.items())
    findings = hard_findings(draft, sections, Constraints(), [])
    assert findings and all(secs == ["Analysis"] for _, secs in findings)


def test_critic_log_records_what_was_wrong_not_just_how_many():
    llm = FakeLLM(script(**{"writer:Introduction": prose("[S99]")}))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert any("S99" in line for line in out["log"] if line.startswith("critic:"))


def test_llm_critic_reviews_the_rendered_paper_not_the_internal_marker_draft():
    llm = FakeLLM(script())
    run(llm, decisions=[approve(), approve()])
    seen = llm.prompts["critic"]
    assert "[S1]" not in seen and "(B, 2022)" in seen  # reader-facing (Author, Year) citations
    assert "Counted body length" in seen



def test_critic_prompt_omits_reference_list_and_uncited_sources():
    llm = FakeLLM(script(**{"writer:Analysis": prose("[S2]")}))  # S3 is a source but is never cited
    run(llm, decisions=[approve(), approve()])
    seen = llm.prompts["critic"]
    assert "## References" not in seen
    assert 'id="S1"' in seen and 'id="S2"' in seen and 'id="S3"' not in seen


def test_typography_is_normalised_to_plain_ascii():
    from app.checks import clean_typography
    assert clean_typography("last‑mile delivery‐cost\u200b") == "last-mile delivery-cost"


def test_writer_output_is_typography_cleaned():
    llm = FakeLLM(script(**{"writer:Introduction": prose("last‑mile [S1]")}))
    _, _, out = run(llm, decisions=[approve(), approve()])
    assert "‑" not in out["draft"] and "last-mile" in out["sections"]["Introduction"]


def test_content_rewrite_preserves_the_sections_current_length():
    from app.checks import word_count
    original = prose("[S2] and [S3]", 200)
    llm = FakeLLM(script(
        **{"writer:Analysis": [original, prose("[S2] and [S3]", 190)]},
        critic=[{"approved": False, "issues": ["unsupported claim"], "sections_to_fix": ["Analysis"]},
                {"approved": True}],
    ))
    run(llm, decisions=[approve(), approve()])
    expected = round(word_count(original) * 0.95)
    assert f"about {expected} words" in llm.systems["writer:Analysis"]  # not the planner's 50-word target
