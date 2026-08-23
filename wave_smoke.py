"""Offline exercise of the wave loop — no API key, no database, no cost.

    python wave_smoke.py

Stubs the four seams the loop touches (model calls, instrument dispatch,
embeddings, and the synthesis call) and drives investigate() through the paths
that are easy to break and expensive to discover in production: escalation,
the barrier surviving a dead worker, budget exhaustion, citation validation,
dedup, and the wave-1 fallback.

Not an eval harness. It says nothing about answer quality — that needs the
synthetic corpus loop_spec.md §9 calls for. What it protects is the control
flow: that a wave still ends, merges, and reports honestly when something in
it goes wrong.

Named to stay out of the way of test.py, which is the multi-bot Discord
conversation service and not a test at all.
"""

import asyncio
import hashlib
import json
import logging
import math
import types as pytypes

import db
import loop
import retrieval
from prompts import GRADER_PROMPT, PLANNER_PROMPT, SUFFICIENCY_PROMPT


# --- fake Gemini responses ---------------------------------------------------

class FakePart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeResponse:
    def __init__(self, parts, tokens=100):
        self.candidates = [pytypes.SimpleNamespace(
            content=pytypes.SimpleNamespace(parts=parts))]
        self.usage_metadata = pytypes.SimpleNamespace(
            total_token_count=tokens, prompt_token_count=int(tokens * 0.8))


def text_response(obj):
    return FakeResponse([FakePart(text=json.dumps(obj))])


def tool_response(calls):
    return FakeResponse([FakePart(function_call=FakeCall(n, a)) for n, a in calls])


async def fake_embed(texts, *, is_query=False):
    """Deterministic unit vectors: identical text embeds identically, so the
    dedup path is exercised for real, and unrelated text lands far apart."""
    out = []
    for t in texts:
        digest = hashlib.sha256(t.encode()).digest()
        vec = [b / 255.0 for b in digest[:16]]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


ROWS = [
    {"discord_message_id": 1401, "channel_id": 9, "author_id": 77,
     "content": "moving my brother saturday, can't do the cabin",
     "created_at": "2026-03-01T10:00"},
    {"discord_message_id": 1402, "channel_id": 9, "author_id": 88,
     "content": "so we push it to the 17th?", "created_at": "2026-03-01T10:02"},
]


class Scenario:
    """Decides what each role replies, and records who was called."""

    def __init__(self, *, plan, grade, sufficient, tool_calls=None,
                 worker_delay=0.0, worker_raises=False):
        self.plan = plan              # one payload per wave, last one repeats
        self.grade = grade            # consumed in order, last one repeats
        self.sufficient = sufficient
        self.tool_calls = tool_calls or [("keyword_search", {"term": "cabin"})]
        self.worker_delay = worker_delay
        self.worker_raises = worker_raises
        self.calls = []
        self._i = {"plan": 0, "grade": 0, "suff": 0}

    def _next(self, key, seq):
        payload = seq[min(self._i[key], len(seq) - 1)]
        self._i[key] += 1
        return payload

    async def call_model(self, *, model, contents, budget, label="call", system=None,
                         tools=False, cache_name=None, json_out=False,
                         deterministic=False):
        # Role is derived from the prompt object, not the label, so a call site
        # that gets relabelled but sends the wrong prompt still fails here.
        role = ("worker" if tools else
                "planner" if system is PLANNER_PROMPT else
                "grader" if system is GRADER_PROMPT else
                "sufficiency" if system is SUFFICIENCY_PROMPT else "unknown")
        assert label in (role, "call"), f"call labelled {label!r} sent the {role} prompt"
        self.calls.append((role, model))

        if role == "worker":
            if self.worker_delay:
                await asyncio.sleep(self.worker_delay)
            if self.worker_raises:
                raise RuntimeError("worker blew up")
            response = tool_response(self.tool_calls)
        elif role == "planner":
            response = text_response(self._next("plan", self.plan))
        elif role == "grader":
            response = text_response(self._next("grade", self.grade))
        elif role == "sufficiency":
            response = text_response(self._next("suff", self.sufficient))
        else:
            raise AssertionError(f"unrecognized system prompt: {str(system)[:60]!r}")

        budget.record(response)
        return response

    async def execute_call(self, name, args):
        return list(ROWS)

    def count(self, role):
        return sum(1 for r, _ in self.calls if r == role)

    def models_for(self, role):
        return [m for r, m in self.calls if r == role]


def install(scenario):
    """Point the loop's four outside edges at the scenario."""
    loop._call_model = scenario.call_model
    retrieval.execute_call = scenario.execute_call
    loop.embed_texts = fake_embed
    loop._worker_cache = lambda model: asyncio.sleep(0, result=None)
    db.open_pool = lambda: asyncio.sleep(0)
    loop.ask_gemini = lambda *a, **k: asyncio.sleep(0, result="ANSWER FROM LEDGER")


# --- payloads ----------------------------------------------------------------

PLAN_TWO = {"sub_questions": [
    {"sub_question": "when was the cabin trip moved to", "priority": 1,
     "rationale": "the date is the crux", "expected_answer_type": "a date"},
    {"sub_question": "who objected to the original date", "priority": 2,
     "rationale": "who drove the change", "expected_answer_type": "a person"},
], "note": "split on date and driver"}

GRADE_RESOLVED = {
    "graded": [{"message_id": "1401", "score": 0.9, "why": "states the conflict"},
               {"message_id": "1402", "score": 0.8, "why": "proposes the new date"}],
    "facts": [{"claim": "the cabin trip moved to the 17th",
               "citations": {"1402": "so we push it to the 17th?"}}],
    "inferences": [{"claim": "the move was driven by the brother's move",
                    "based_on": ["F1"], "competing": ["scheduling coincidence"]}],
    "status": "resolved", "gap": "", "ambiguous": False, "note": "found the date",
}

SUFF_YES = {"sufficient": "yes", "gaps": [], "note": "ledger answers it"}


# --- cases -------------------------------------------------------------------

async def case_happy():
    s = Scenario(plan=[PLAN_TWO], grade=[GRADE_RESOLVED], sufficient=[SUFF_YES])
    install(s)
    run = await loop.investigate("why did the cabin trip move?")
    assert run.verdict == "yes", run.verdict
    assert run.waves == 1
    assert len(run.ledger.facts) == 2 and run.ledger.facts[0]["id"] == "F1"
    assert run.ledger.message_ids == {"1401", "1402"}, run.ledger.message_ids
    assert (s.count("planner"), s.count("worker"), s.count("grader")) == (1, 2, 2)
    assert s.count("sufficiency") == 0, "cheap gate should have skipped the model"
    assert run.llm_calls == 5, run.llm_calls
    return f"verdict={run.verdict} waves={run.waves} facts={len(run.ledger.facts)} calls={run.llm_calls}"


async def case_two_waves():
    refine = dict(GRADE_RESOLVED, status="refine", gap="whether Sam agreed",
                  graded=[{"message_id": "1401", "score": 0.3, "why": "weak"}])
    wave2 = {"sub_questions": [{"sub_question": "did Sam agree to the 17th",
                                "priority": 1, "rationale": "the gap",
                                "expected_answer_type": "yes/no"}], "note": "chase it"}
    s = Scenario(plan=[PLAN_TWO, wave2],
                 grade=[refine] * 6 + [GRADE_RESOLVED],
                 sufficient=[{"sufficient": "no", "gaps": ["whether Sam agreed"],
                              "note": "one gap"}, SUFF_YES])
    install(s)
    run = await loop.investigate("why did the cabin trip move?")
    assert run.waves == 2 and run.verdict == "yes"
    # A 0.3 top score is under escalate_score, so the second round goes large.
    assert loop.LARGE_MODEL in s.models_for("worker")
    assert run.trajectory[0]["results"][0]["escalated"] is True
    assert run.trajectory[1]["sub_questions"] == ["did Sam agree to the 17th"]
    return f"waves={run.waves} escalated=True calls={run.llm_calls}"


async def case_fallback():
    s = Scenario(plan=[{"sub_questions": [], "note": "cannot split"}],
                 grade=[GRADE_RESOLVED], sufficient=[SUFF_YES])
    install(s)
    run = await loop.investigate("what happened?")
    assert run.trajectory[0]["sub_questions"] == ["what happened?"]
    assert s.count("worker") == 1
    return "wave 1 fell back to the original question as one sub-question"


async def case_stalled():
    s = Scenario(plan=[PLAN_TWO, {"sub_questions": [], "note": "nothing left"}],
                 grade=[dict(GRADE_RESOLVED, status="refine", gap="x")],
                 sufficient=[{"sufficient": "no", "gaps": ["x"], "note": "more"}])
    install(s)
    run = await loop.investigate("q")
    assert run.verdict == "stalled", run.verdict
    return f"verdict={run.verdict} after {run.waves} waves"


async def case_timeout():
    s = Scenario(plan=[PLAN_TWO], grade=[GRADE_RESOLVED], sufficient=[SUFF_YES],
                 worker_delay=0.30)
    install(s)
    run = await loop.investigate("q", cfg=loop.WaveConfig(worker_timeout_s=0.05))
    statuses = [r["status"] for r in run.trajectory[0]["results"]]
    assert statuses == ["timeout", "timeout"], statuses
    assert run.ledger.facts == []
    return f"both workers cancelled at the barrier, wave still closed"


async def case_worker_error():
    s = Scenario(plan=[PLAN_TWO], grade=[GRADE_RESOLVED],
                 sufficient=[{"sufficient": "unanswerable", "gaps": [],
                              "note": "nothing came back"}],
                 worker_raises=True)
    install(s)
    run = await loop.investigate("q")
    assert [r["status"] for r in run.trajectory[0]["results"]] == ["error", "error"]
    assert run.verdict == "unanswerable"
    return f"worker exceptions became gaps, verdict={run.verdict}"


async def case_worker_error_deterministic():
    # The sequential path has to fail the same way the parallel one does.
    s = Scenario(plan=[PLAN_TWO], grade=[GRADE_RESOLVED],
                 sufficient=[{"sufficient": "unanswerable", "gaps": [], "note": ""}],
                 worker_raises=True)
    install(s)
    run = await loop.investigate("q", cfg=loop.WaveConfig(deterministic=True))
    assert [r["status"] for r in run.trajectory[0]["results"]] == ["error", "error"]
    return "sequential path survived a worker exception too"


async def case_budget():
    s = Scenario(plan=[PLAN_TWO],
                 grade=[dict(GRADE_RESOLVED, status="refine", gap="g")],
                 sufficient=[{"sufficient": "no", "gaps": ["g"], "note": "more"}])
    install(s)
    run = await loop.investigate("q", cfg=loop.WaveConfig(max_llm_calls=4, max_waves=3))
    assert run.verdict == "budget_exhausted", run.verdict
    assert run.llm_calls <= 8, run.llm_calls
    return f"stopped at {run.llm_calls} calls against a cap of 4, verdict={run.verdict}"


async def case_bad_citations():
    bad = dict(GRADE_RESOLVED, facts=[
        {"claim": "no citations here"},
        {"claim": "empty dict", "citations": {}},
        {"claim": "good one", "citations": {"1401": "moving my brother"}},
    ])
    s = Scenario(plan=[{"sub_questions": [{"sub_question": "x", "priority": 1}],
                        "note": ""}],
                 grade=[bad], sufficient=[SUFF_YES])
    install(s)
    run = await loop.investigate("q")
    claims = [f["claim"] for f in run.ledger.facts]
    assert claims == ["good one"], claims
    return "kept the cited fact, rejected the two without citations"


async def case_dedup():
    dupe = {"sub_questions": [
        {"sub_question": "when was the cabin trip moved to", "priority": 1},
        {"sub_question": "when was the cabin trip moved to", "priority": 2},
    ], "note": "accidental repeat"}
    s = Scenario(plan=[dupe], grade=[GRADE_RESOLVED], sufficient=[SUFF_YES])
    install(s)
    run = await loop.investigate("q")
    assert s.count("worker") == 1
    assert run.trajectory[0]["deduped"] == ["when was the cabin trip moved to"]
    return "two identical sub-questions became one worker"


async def case_answer_seam():
    s = Scenario(plan=[PLAN_TWO], grade=[GRADE_RESOLVED], sufficient=[SUFF_YES])
    install(s)
    text = await loop.answer("why did the cabin trip move?")
    assert text == "ANSWER FROM LEDGER", text
    return "bot.py's seam still returns synthesized text"


CASES = [
    ("happy path", case_happy),
    ("two waves + escalation", case_two_waves),
    ("wave-1 fallback", case_fallback),
    ("stalled", case_stalled),
    ("worker timeout", case_timeout),
    ("worker exception", case_worker_error),
    ("worker exception (deterministic)", case_worker_error_deterministic),
    ("budget exhaustion", case_budget),
    ("citation validation", case_bad_citations),
    ("sub-question dedup", case_dedup),
    ("answer() seam", case_answer_seam),
]


async def main():
    print("=" * 72)
    for name, case in CASES:
        # The loop logs every wave in full, which is the point in production and
        # noise here; only the assertions matter.
        logging.disable(logging.CRITICAL)
        try:
            detail = await case()
        finally:
            logging.disable(logging.NOTSET)
        print(f"  ok  {name:34} {detail}")
    print("=" * 72)
    print(f"all {len(CASES)} cases passed")


if __name__ == "__main__":
    asyncio.run(main())
