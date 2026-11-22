"""Qualification-aware automatic routing (opt-in: runtime.yaml  routing.mode == "qualified").

Every 'auto' request is (1) classified into a task category, (2) matched against a qualification table that
comes from measured evidence (config, not code), (3) assigned to the best AVAILABLE model of the best tier:
qualified > limited > best_effort, weighing residency, queue length and recent load failures. Responses say
whether the chosen model is certified for the category and whether the answer must be verified.
Explicit model selection is never changed; it is only annotated. Models listed under routing.experimental
(deepseek) are never chosen automatically."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import MODEL_ROSTER, resolve_model_id
from .router import Decision, Router

# (regex, weight). score(category) = sum(weight * min(hits, 2)); highest wins if >= MIN_SCORE.
_RULES = {
    "test_generation": [
        (r"\b(write|generate|create|add|produce|draft|give me)\b[^.\n]{0,50}\b(unit |integration |regression |pytest )?tests?\b", 6),
        (r"\btest (cases?|suite|plan|coverage)\b", 5), (r"\b(pytest|unittest|jest|junit)\b", 4),
        (r"\bassert statements?\b", 4)],
    "code_review": [
        (r"\bcode review\b|\breview (this|the|my|following)\b|\bpull request\b", 6), (r"\b(audit|critique|inspect)\b", 4),
        (r"\b(defect|vulnerabilit(y|ies)|security (issue|risk|problem|flaw))\b", 5),
        (r"\bwhat'?s wrong with\b|\bany (bugs?|issues?|problems?)\b", 3)],
    "debugging": [
        (r"\bdebug(ging)?\b|\btraceback\b|\bstack ?trace\b", 6), (r"\b(has|have|contains?|there'?s|found) an? (bug|defect|error|mistake)\b|\bfind (and fix )?the (bug|error|mistake)\b|\bbuggy\b|\bbug\b", 9),
        (r"\b(fix|repair|correct)\b[^.\n]{0,30}\b(bug|error|crash|failure|issue)\b", 6),
        (r"\b(KeyError|IndexError|TypeError|ValueError|NameError|AttributeError|segfault)\b", 4),
        (r"\b(fails?|failing|crash(es|ed)?)\b", 3),
        (r"\bdoes(n'?t| not) (work|return|produce)\b|\bnot working\b|\bwrong (output|result)\b", 4),
        (r"\b(infinite loop|never (terminates|ends|stops|finishes)|hangs?|deadlock|memory leak|off-by-one)\b|\bwhat('?s| is) wrong\b", 5)],
    "documentation": [
        (r"\bdocstrings?\b|\bdocumentation\b|\b(google|numpy|sphinx)[- ]style\b|\bdocument (this|the)\b|\breadme\b|\bapi docs?\b|\bjsdoc\b|\bchangelog\b", 6),
        (r"\bwrite (the )?docs\b|\badd comments\b", 4)],
    "architecture": [
        (r"\barchitect(ure|ural)?\b|\bsystem design\b|\bdesign (a|an|the)\b[^.\n]{0,40}\b(system|service|pipeline|platform|schema|api)\b", 6),
        (r"\btrade-?offs?\b|\bpros and cons\b|\bshould we\b|\bcompare\b[^.\n]{0,40}\b(with|vs\.?|versus|and)\b", 4),
        (r"\b(scal(e|ing|ability)|high availability|multi-?tenan\w*|microservices?|monolith|zero-downtime|disaster recovery|rate limiter|observability|idempoten\w*)", 4),
        (r"\bstrateg(y|ic)\b|\broadmap\b", 2),
        (r"\bdesign\b[^.\n]{0,60}\b(authentication|authorization|backup|caching|storage|database|deployment|queue|architecture)\b", 6),
        (r"\b(backup|disaster[- ]recovery|caching|cache invalidation|stampede|failover|sharding|replication|load balanc\w+|event[- ]driven|consistency)\b", 4),
        (r"\bexplain the (main )?(risks?|trade-?offs?|failure modes?|design|approach)\b|\bhow would you (design|structure|architect)\b", 5)],
    "reasoning": [
        (r"\bstep by step\b|\bthink (through|carefully)\b|\bprove\b", 4),
        (r"\bhow many\b|\bwhat is the (sum|product|probability|total|distance|value|answer)\b|\bcalculate\b|\bsolve\b|\bprobability\b", 4),
        (r"\b(riddle|puzzle)\b|\bwhat is \(?[a-z]\)?\s*\?|=\s*-?\d+", 4),
        (r"\b(taller|shorter|older|younger|heavier|lighter|faster|slower) than\b|\bwhat day (of the week )?will it be\b|\bdays? from (today|now)\b|\bdoubles? (in size )?every\b|\bhow (long|many|much|far|old)\b[^.\n]{0,40}\?|\bwho is the (tallest|shortest|oldest|youngest|fastest)\b", 5)],
    "coding": [
        (r"\b(write|implement|create|build|develop|generate|complete)\b[^.\n]{0,40}\b(function|class|method|script|program|module|endpoint|algorithm|query|snippet|cli|component)\b", 6),
        (r"\bimplement\b|\brefactor\b|\brewrite\b", 4), (r"\bdef \w+\(|\bclass \w+|#include|\bfn main\b", 2)],
    "triage": [
        (r"\bcommit message\b|\bgit (commit|log|diff|rebase|stash|status|branch)\b", 6),
        (r"\bone[- ]?liner\b|\bregex\b|\bregular expression\b|\bsed\b|\bawk\b|\bgrep\b|\bbash\b|\bshell (command|script)\b|\bpowershell\b", 5),
        (r"\b(quick(ly)?|rename|typo|lint)\b|\bsyntax check\b", 3), (r"\btriage\b|\bone[- ]line (summary|answer|reply)\b|\bin one line\b", 5), (r"\bsummari[sz]e (this |the )?(diff|log|output|error)\b", 4)],
}
_PRIORITY = ["test_generation", "code_review", "debugging", "documentation", "architecture", "reasoning", "coding", "triage"]
_COMPILED = {c: [(re.compile(p, re.I), w) for p, w in rs] for c, rs in _RULES.items()}
MIN_SCORE = 4
LONG_CONTEXT_TOKENS = 1500          # estimated prompt tokens above which an unclassified request counts as long-context
TIERS = ("qualified", "limited", "best_effort")


class NoQualifiedModel(Exception):
    """require_certified was requested but no certified model is available."""


class NoSuitableModel(Exception):
    """no candidate model is available for the category."""


def classify(text: str, body: dict | None = None) -> tuple[str, dict]:
    body = body or {}
    hint = (body.get("maple") or {}).get("category") if isinstance(body.get("maple"), dict) else None
    if hint:
        return str(hint), {"source": "caller hint"}
    if body.get("tools") or body.get("response_format") or body.get("tool_choice"):
        return "api_tool", {"source": "tools/response_format present"}
    t = text or ""
    prose = re.sub(r"```.*?```", " <code> ", t, flags=re.S)             # code inside the prompt is data, not the task
    t_eff = prose if len(prose.strip()) >= 20 else t
    sample = t_eff if len(t_eff) <= 3000 else t_eff[:600] + "\n" + t_eff[-2400:]      # the question is usually at the end of long prompts
    scores = {c: sum(w * min(len(rx.findall(sample)), 2) for rx, w in rules) for c, rules in _COMPILED.items()}
    best = max(scores.values())
    if best >= MIN_SCORE:
        cat = next(c for c in _PRIORITY if scores[c] == best)
        return cat, {"source": "keywords", "scores": {k: v for k, v in scores.items() if v}}
    if len(t) / 3.5 >= LONG_CONTEXT_TOKENS:
        return "long_context", {"source": "long prompt, no task signal"}
    return "general", {"source": "no task signal"}


@dataclass
class QDecision(Decision):
    category: str = "general"
    tier: str = "best_effort"
    certified: bool = False
    verify_required: bool = False
    chain: list = field(default_factory=list)       # [(model, tier)] in preference order, includes the first choice
    info: dict = field(default_factory=dict)

    def chain_decisions(self):
        out = []
        for n, (mid, tier) in enumerate(self.chain):
            if n == 0:
                out.append(self)
                continue
            out.append(QDecision(mid, "fallback from %s: %s" % (self.model, self.reason), {}, False, self.category, tier,
                                 tier == "qualified", self._verify(tier), [], dict(self.info)))
        return out

    def _verify(self, tier):
        return bool(self.info.get("verify_default")) or tier != "qualified"

    def public(self):
        return {"category": self.category, "tier": self.tier, "certified": self.certified,
                "verify_required": self.verify_required, "fallback_chain": [m for m, _ in self.chain], **{
                    k: v for k, v in self.info.items() if k in ("source", "costs", "note")}}

    def headers(self):
        return {"X-MAPLE-Category": self.category, "X-MAPLE-Tier": self.tier,
                "X-MAPLE-Certified": "true" if self.certified else "false",
                "X-MAPLE-Verify": "required" if self.verify_required else "recommended-not-required"}


class QualifiedRouter(Router):
    def __init__(self, default, roster, cfg, manager):
        super().__init__(default=default, roster=roster)
        rc = cfg.get("routing") or {}
        self.table = rc.get("qualification", {})
        self.experimental = set(rc.get("experimental", ["deepseek"]))
        self.w = {"rank": 0.5, "queue": 1.0, "not_resident": 2.0, **(rc.get("weights") or {})}
        self.manager = manager

    # -- availability ------------------------------------------------------------------------------------------
    def _state(self):
        snap = getattr(self.manager, "snapshot", None)
        return snap() if callable(snap) else {"resident": {}, "failed": {}}

    def _cost(self, mid, rank, st):
        r = st["resident"].get(mid)
        q = (1 if r and r.get("busy") else 0) + (r.get("waiting", 0) if r else st.get("waiting", {}).get(mid, 0))
        return round(self.w["rank"] * rank + self.w["queue"] * q + (0 if r else self.w["not_resident"]), 2)

    def _candidates(self, category, exclude, require_certified, st):
        entry = self.table.get(category) or self.table.get("general") or {}
        ordered, seen = [], set()
        for tier in TIERS:
            if require_certified and tier != "qualified":
                break
            for rank, mid in enumerate(entry.get(tier, [])):
                if mid in seen or mid in exclude or mid in self.experimental or mid not in self.roster:
                    continue
                seen.add(mid)
                if mid in st["failed"]:          # recent load failure: still cooling down
                    continue
                ordered.append((tier, rank, mid))
        chain, costs = [], {}
        for tier in TIERS:
            part = sorted(((self._cost(m, rk, st), rk, m) for t, rk, m in ordered if t == tier))
            for c, rk, m in part:
                chain.append((m, tier)); costs[m] = c
        return chain, costs, entry

    # -- selection ---------------------------------------------------------------------------------------------
    def select(self, text, requested=None, body=None):
        body = body or {}
        mp = body.get("maple") if isinstance(body.get("maple"), dict) else {}
        category, cinfo = classify(text, body)
        entry = self.table.get(category) or self.table.get("general") or {}
        if requested and requested.strip().lower() not in ("auto", "maple", "maple-auto", ""):
            d = super().select(text, requested)               # explicit selection is never changed, only annotated
            tier = next((t for t in TIERS if d.model in entry.get(t, [])), "uncertified")
            return QDecision(d.model, d.reason, {}, True, category, tier, tier == "qualified",
                             bool(entry.get("verify_required")) or tier != "qualified", [],
                             {**cinfo, "verify_default": bool(entry.get("verify_required"))})
        exclude = {resolve_model_id(x) for x in (mp.get("exclude") or []) if x in self.roster or x in ("quant",)}
        require = bool(mp.get("require_certified"))
        st = self._state()
        chain, costs, entry = self._candidates(category, exclude, require, st)
        if not chain:
            if require:
                raise NoQualifiedModel("no certified model available for category '%s' (excluded: %s)" % (category, sorted(exclude) or "none"))
            raise NoSuitableModel("no available model for category '%s'" % category)
        mid, tier = chain[0]
        note = {"qualified": "qualified for '%s'" % category,
                "limited": "only a LIMITED-rated model is available for '%s'" % category,
                "best_effort": "no model is certified for '%s'; best-effort choice" % category}[tier]
        verify = bool(entry.get("verify_required")) or tier != "qualified"
        return QDecision(mid, "%s: %s (cost %.1f)" % (category, note, costs[mid]), {}, False, category, tier, tier == "qualified",
                         verify, chain, {**cinfo, "costs": costs, "note": note, "verify_default": bool(entry.get("verify_required"))})


def make_router(cfg, roster, manager):
    rc = cfg.get("routing") or {}
    if rc.get("mode") == "qualified" and callable(getattr(manager, "snapshot", None)):
        return QualifiedRouter(cfg.get("default_model", "gemma"), roster, cfg, manager)
    return Router(default=cfg.get("default_model", "gemma"))
