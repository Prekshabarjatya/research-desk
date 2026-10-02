"""Local UI development server. Runs the real API, worker and graph with a scripted model and
canned sources, so every screen can be exercised with no API key, no database and no network.

    .venv/bin/python scripts/dev_server.py          # scripted fixture, http://127.0.0.1:8765, token: dev
    .venv/bin/python scripts/dev_server.py --live   # real OpenRouter model + real source search (needs .env)

Keywords in the assignment pick a scenario: "fail" stops the run at the planner (as a bad key
would), "review" makes the reviewer reject every draft until the revision cap. Anything else
approves. All text here is fixture output, not research, and is labelled as such."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from langgraph.checkpoint.memory import MemorySaver

from app.api import create_app
from app.config import settings
from app.graph import build_graph
from app.llm import Completion, LLMError
from app.models import Source
from app.nodes import Tools
from app.store import MemoryRunStore
from app.worker import run_forever

TOPIC = "[Dev fixture] AI route optimization and last-mile delivery costs in urban e-commerce"
SOURCES = [
    ("Optimization and Machine Learning Applied to Last-Mile Logistics: A Review", ["Nadia Giuffrida", "Jenny Fajardo Calderin", "Antonio D. Masegosa"], 2022, "10.3390/su14095329", "Sustainability"),
    ("Data-driven optimization for last-mile delivery", ["Hongrui Chu", "Wensi Zhang", "Pengfei Bai", "Yahong Chen"], 2021, "10.1007/s40747-021-00293-1", "Complex & Intelligent Systems"),
    ("The last mile challenge: evaluating the effects of customer density and delivery window patterns", ["Kenneth K. Boyer", "Andrea M. Prud'homme", "Wenming Chung"], 2009, "10.1002/j.2158-1592.2009.tb00104.x", "Journal of Business Logistics"),
    ("Last mile logistics: Research trends and needs", ["Emrah Demir", "Aris Syntetos", "Tom Van Woensel"], 2022, "10.1093/imaman/dpac006", "IMA Journal of Management Mathematics"),
    ("A Review of Last-Mile Delivery Optimization: Strategies, Technologies, Drone Integration, and Future Trends", ["Abdullahi Sani Shuaibu", "Ashraf Mahmoud", "Tarek Sheltami"], 2025, "10.3390/drones9030158", "Drones"),
    ("Artificial Intelligence in Logistics Optimization with Sustainable Criteria: A Review", ["Wenrui Chen", "Yingying Men", "Nuria Fuster", "Carlos Osorio", "Angel A. Juan"], 2024, "10.3390/su16219145", "Sustainability"),
]
PROSE = {
    "Introduction": "Last-mile delivery is the most expensive leg of an e-commerce order, and cities make it harder through congestion, parking limits and short delivery windows [S1]. Routing software that learns from past deliveries is often proposed as a way to lower that cost without adding vehicles [S2]. This paper asks how far that promise holds in dense urban settings, and where the evidence is thin. It argues that learning-based routing lowers cost mainly by raising vehicle utilization, and that gains shrink once customer density falls [S3].",
    "Literature Review": "Reviews of the field separate classical optimization from machine learning approaches and note that the two are increasingly combined [S1]. Early work focused on how customer density and delivery windows shape route length, which remains the baseline against which newer methods are judged [S3]. Later studies use historical delivery data to predict travel times and service durations, then feed those predictions into the routing step [S2]. More recent surveys widen the scope to include drones and other vehicle types, and point out that evaluation methods differ so much that headline savings are hard to compare across studies [S5][S6].",
    "Analysis": "Across the sources, the strongest and most consistent mechanism is better use of vehicle capacity: fewer partly empty trips and shorter total distance [S2]. Density matters because consolidation is easier when many stops sit close together, which favors urban cores over rural areas [S3]. Sustainability reviews add that lower distance also reduces emissions, though they caution that rebound effects can offset part of the gain when faster delivery raises order volume [S6]. The evidence for real-time re-routing is weaker, mostly because few studies test it outside simulation [S4].",
    "Conclusion": "Learning-based route optimization can reduce last-mile cost in dense urban delivery, mainly through higher vehicle utilization [S2][S3]. The size of the benefit depends on customer density and on how the study was evaluated, so single savings figures should be read with care [S5]. Future work would benefit from shared benchmarks and field trials rather than simulation alone [S4].",
}


class DevLLM:
    """Implements the LLM protocol with scripted output and a small delay per call."""

    def __init__(self, delay=0.9):
        self.delay = delay
        self.scenario = ""

    def _wait(self):
        time.sleep(self.delay)

    def complete(self, system, user, *, tier="fast", purpose=""):
        self._wait()
        if purpose == "thesis":
            return Completion("Learning-based route optimization lowers urban last-mile cost mainly by raising vehicle utilization, with benefits that fall as customer density falls.", 420)
        if purpose.startswith("writer:"):
            title = purpose.split(":", 1)[1]
            return Completion(PROSE.get(title, PROSE["Analysis"]), 610)
        raise AssertionError(purpose)

    def complete_json(self, system, user, schema, *, tier="fast", purpose=""):
        self._wait()
        if purpose == "analyst":
            self.scenario = "fail" if "fail" in user.lower() else "review" if "review" in user.lower() else ""
            return schema.model_validate({"min_words": 0, "max_words": 0, "citation_style": "APA",
                                          "required_sections": ["Introduction", "Conclusion"], "goals": []}), 300
        if purpose == "strategist":
            return schema.model_validate({"topic": TOPIC, "rationale": "Narrow, arguable, well covered.",
                                          "search_queries": ["AI route optimization last-mile delivery cost",
                                                             "machine learning vehicle routing urban logistics",
                                                             "customer density delivery window last mile"]}), 350
        if purpose == "planner":
            if self.scenario == "fail":
                raise LLMError("All providers failed for 'planner': [\"openrouter: Error code: 401 - {'error': {'message': 'Invalid API Key'}}\"]")
            plan = [("Introduction", "Frame the cost problem", ["S1", "S2", "S3"]),
                    ("Literature Review", "Survey the approaches", ["S1", "S2", "S3", "S5", "S6"]),
                    ("Analysis", "Weigh the evidence", ["S2", "S3", "S4", "S6"]),
                    ("Conclusion", "State what holds", ["S2", "S3", "S4", "S5"])]
            return schema.model_validate({"sections": [{"title": t, "goal": g, "source_ids": ids, "target_words": 300} for t, g, ids in plan]}), 500
        if purpose == "critic":
            if self.scenario == "review":
                return schema.model_validate({"approved": False, "issues": [
                    "The claim about rebound effects rests on a single review and needs a second source.",
                    "The link between customer density and cost is stated more strongly than the cited evidence supports."],
                    "sections_to_fix": ["Analysis"]}), 700
            return schema.model_validate({"approved": True}), 700
        raise AssertionError(purpose)


def build_live():
    """Real model and real literature search, with in-memory state (nothing persists on restart)."""
    import httpx

    from app.citations import verify_source
    from app.literature import search_all
    from app.llm import RoutedLLM, build_providers

    http = httpx.Client(headers={"User-Agent": "research-paper-agents/0.1"})
    tools = Tools(search=lambda q: search_all(q, http), verify=lambda s: verify_source(s, http))
    return RoutedLLM(build_providers()), tools


def main():
    settings.api_token = "dev"
    settings.worker_poll_seconds = 0.3
    if "--live" in sys.argv:
        llm, tools = build_live()
        store = MemoryRunStore()
        graph = build_graph(llm, tools, MemorySaver())
        threading.Thread(target=run_forever, args=(store, graph), daemon=True).start()
        print("LIVE dev server: http://127.0.0.1:8765  token: dev  (real OpenRouter model + real sources, uses your quota)")
        uvicorn.run(create_app(store), host="127.0.0.1", port=8765, log_level="warning")
        return
    settings.max_revisions = 2
    pool = [Source(title=t, authors=a, year=y, doi=d, venue=v, abstract=f"{t}. Sample abstract for the dev fixture.") for t, a, y, d, v in SOURCES]
    tools = Tools(search=lambda q: [s.model_copy() for s in pool], verify=lambda s: s.model_copy(update={"verified": True}))
    store = MemoryRunStore()
    graph = build_graph(DevLLM(), tools, MemorySaver())
    threading.Thread(target=run_forever, args=(store, graph), daemon=True).start()
    print("Dev fixture server: http://127.0.0.1:8765  token: dev  (scripted model, sample text)")
    uvicorn.run(create_app(store), host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
