"""Automatic specialist selection: transparent keyword/pattern scoring.

Deliberately simple and explainable; explicit model selection always wins."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import MODEL_ROSTER, resolve_model_id

# (pattern, weight). Patterns are matched case-insensitively on the prompt text.
_RULES = {
    "deepseek": [
        (r"\bc\+\+\d*\b|\bcpp\b", 4), (r"\brust\b|\bcargo\b|\bborrow checker\b", 4),
        (r"\bcuda\b|\bgpu kernel\b|\bnvcc\b", 4), (r"\block[- ]free\b|\batomic\b|\bmutex\b|\bconcurren", 4),
        (r"\balgorithm|\bcomplexity\b|\bbig[- ]o\b|\bdata structure", 3),
        (r"\bpointer\b|\bmemory (safety|leak|layout)|\bsegfault\b|\bsimd\b|\bkernel\b", 3),
        (r"\bsystems? programming\b|\bsyscall\b|\bthread", 2), (r"#include|std::|fn main|unsafe \{", 4),
    ],
    "gemma": [
        (r"\bpython\b|\bpip\b|\bdjango\b|\bflask\b|\bfastapi\b|\bpandas\b", 4),
        (r"\bhtml\b|\bcss\b|\bjavascript\b|\btypescript\b|\breact\b|\bvue\b|\bfrontend\b|\bweb ?(site|app|dev)", 4),
        (r"\brefactor|\bclean ?up\b|\bcode review\b|\breview (this|my)\b", 4),
        (r"\bdocument(ation)?\b|\bdocstring\b|\breadme\b", 3), (r"\bdef \w+\(|import \w+|<div|</", 3),
    ],
    "qwen3": [
        (r"\btrading\b|\bquant\b|\bportfolio\b|\bhedge\b|\bsharpe\b|\bbacktest|\balpha\b|\bvolatility\b", 5),
        (r"\brisk\b|\bdrawdown\b|\bposition siz", 4), (r"\barchitect(ure|ural)?\b|\bsystem design\b|\btrade-?offs?\b", 4),
        (r"\bstrateg(y|ic)\b|\bplan(ning)?\b|\bdecision\b|\bshould we\b|\bpros and cons\b", 3),
        (r"\breason(ing)? through\b|\bstep by step\b|\bprove\b|\bwhy\b", 2),
    ],
    "fast": [
        (r"\bgit (commit|rebase|log|diff|stash)|\bcommit message\b", 5), (r"\bregex\b|\bregular expression", 5),
        (r"\bone[- ]?liner\b|\bbash\b|\bshell\b|\bsed\b|\bawk\b|\bgrep\b|\bpowershell\b", 4),
        (r"\bquick(ly)?\b|\bsyntax\b|\btypo\b|\bfix (this )?line\b|\bshort\b", 2),
    ],
}

_DEFAULT = "gemma"
_ORDER = ["deepseek", "gemma", "qwen3", "fast"]


@dataclass
class Decision:
    model: str
    reason: str
    scores: dict = field(default_factory=dict)
    explicit: bool = False


class Router:
    def __init__(self, default: str = _DEFAULT, roster=None):
        self.roster = roster or MODEL_ROSTER
        self.default = default
        self._compiled = {m: [(re.compile(p, re.I), w) for p, w in rs] for m, rs in _RULES.items()}

    def score(self, text: str) -> dict:
        scores = {m: 0 for m in self.roster}
        for m, rules in self._compiled.items():
            if m not in scores:
                continue
            for rx, w in rules:
                hits = len(rx.findall(text))
                if hits:
                    scores[m] += w * min(hits, 3)
        return scores

    def select(self, text: str, requested: str | None = None, body: dict | None = None) -> Decision:
        if requested and requested.strip().lower() not in ("auto", "maple", "maple-auto", ""):
            try:
                mid = resolve_model_id(requested)
            except KeyError:
                raise KeyError("unknown model '%s' (valid: %s, auto)" % (requested, ", ".join(self.roster)))
            return Decision(mid, "explicit selection '%s'" % requested, {}, True)
        scores = self.score(text or "")
        best = max(scores.values())
        if best < 3:
            return Decision(self.default, "no strong specialist signal; default model", scores)
        # Tie-break on fixed order; very short quick-task prompts prefer 'fast'.
        top = [m for m in _ORDER if scores.get(m) == best]
        return Decision(top[0], "highest specialist score (%d) among %s" % (best, top), scores)


def last_user_text(messages) -> str:
    texts = []
    for m in messages or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            texts.append(str(c or ""))
    # weight the final user turn but keep earlier context for signals
    return "\n".join(texts[-3:])
