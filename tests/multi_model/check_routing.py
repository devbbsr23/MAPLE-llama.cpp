
import sys, os, threading, time, json
sys.path.insert(0, sys.argv[1])
from maple.qualified_routing import QualifiedRouter, NoQualifiedModel, NoSuitableModel, make_router, classify
from maple.router import Router
from maple.config import MODEL_ROSTER
CFG = json.load(open(sys.argv[2]))
class FM:
    def __init__(s): s.st = {"resident": {m: {"gpu": g, "busy": False, "waiting": 0} for m, g in (("gemma",0),("deepseek",0),("qwen3",1),("fast",1))}, "failed": {}, "waiting": {}}
    def snapshot(s): return s.st
fm = FM(); R = make_router(CFG, MODEL_ROSTER, fm); assert isinstance(R, QualifiedRouter)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)
def sel(text, req=None, **maple): return R.select(text, req, body={"maple": maple} if maple else {})
d = sel("Write a Python function that merges two sorted lists.")
check("coding -> qwen3, certified", d.model == "qwen3" and d.certified and d.tier == "qualified" and not d.verify_required)
d = sel("Write unit tests for a function that parses ISO dates.")
check("test generation -> never certified, verify required (%s/%s)" % (d.model, d.tier), d.tier == "best_effort" and not d.certified and d.verify_required and d.model in ("gemma", "qwen3"))
check("test generation headers say uncertified", d.headers()["X-MAPLE-Certified"] == "false" and d.headers()["X-MAPLE-Verify"] == "required")
try: sel("Write unit tests for a function that parses ISO dates.", require_certified=True); check("require_certified on test generation -> 409", False)
except NoQualifiedModel: check("require_certified on test generation -> NoQualifiedModel", True)
d = sel("Please review this pull request diff for problems.")
check("code review -> gemma certified but verify required", d.model == "gemma" and d.certified and d.verify_required)
d = sel("Design a system that handles 10k requests per second with zero downtime deploys.")
check("architecture -> qwen3", d.model == "qwen3" and d.certified)
d = sel("Add docstrings to every function in this file."); check("documentation -> gemma", d.model == "gemma" and d.certified)
d = sel("Write a commit message for this change."); check("triage -> fast but NOT certified", d.model == "fast" and not d.certified and d.verify_required)
d = sel("Hello, how are you today?"); check("general -> gemma", d.model == "gemma")
d = sel("x", tools=None) if False else R.select("what is the weather", None, body={"tools": [{"type": "function"}]}); check("tools present -> api_tool category", d.category == "api_tool" and d.model == "qwen3")
# queue pressure inside the same tier
fm.st["resident"]["qwen3"].update(busy=True, waiting=3)
d = sel("Find the bug: the function returns the wrong output for negative numbers."); check("debugging: qwen3 busy+3 queued -> gemma (also qualified)", d.model == "gemma" and d.certified)
d = sel("Write a Python function that merges two sorted lists."); check("coding keeps waiting for qualified qwen3 despite queue (no uncertified detour)", d.model == "qwen3" and d.certified)
fm.st["resident"]["qwen3"].update(busy=False, waiting=0)
# failure cooldown
fm.st["failed"]["qwen3"] = 30.0
d = sel("Write a Python function that merges two sorted lists."); check("qwen3 failed -> coding falls to gemma, flagged best_effort/uncertified", d.model == "gemma" and d.tier == "best_effort" and not d.certified and d.verify_required)
try: sel("Write a Python function that merges two sorted lists.", require_certified=True); check("require_certified coding while qwen3 down -> error", False)
except NoQualifiedModel: check("require_certified coding while qwen3 down -> NoQualifiedModel", True)
try: sel("Solve: 3x + 7 = 25. Think step by step."); check("reasoning with qwen3 down -> no unsafe fallback", False)
except NoSuitableModel: check("reasoning with qwen3 down -> NoSuitableModel (503), no unqualified fallback", True)
fm.st["failed"] = {}
# residency preference
del fm.st["resident"]["gemma"]
d = sel("Find the bug in this function: it returns the wrong output."); check("debugging: gemma not resident -> qwen3", d.model == "qwen3")
fm.st["resident"]["gemma"] = {"gpu": 0, "busy": False, "waiting": 0}
# deepseek experimental
for t in ("Write a Python function.", "Review this code for defects.", "Write unit tests for a stack.", "Hello", "commit message please"):
    for ex in ([], ["qwen3"], ["qwen3", "gemma"], ["qwen3", "gemma", "fast"]):
        try: m = sel(t, exclude=ex).model
        except (NoQualifiedModel, NoSuitableModel): continue
        assert m != "deepseek", (t, ex, m)
check("deepseek is never selected automatically", True)
# explicit selection never changed, annotated
d = sel("Write unit tests for a stack.", "deepseek"); check("explicit deepseek honoured, uncertified, verify required", d.model == "deepseek" and d.explicit and not d.certified and d.verify_required)
d = sel("Write unit tests for a stack.", "fast"); check("explicit fast for test generation honoured but uncertified", d.model == "fast" and not d.certified)
d = sel("Write a function.", "quant"); check("explicit alias quant -> qwen3 certified for coding", d.model == "qwen3" and d.certified)
# chain
d = sel("Add docstrings to every function in this file."); ch = d.chain_decisions()
check("documentation chain is gemma, qwen3, fast all qualified", [x.model for x in ch] == ["gemma", "qwen3", "fast"] and all(x.certified for x in ch))
d = sel("Write a Python function that merges two sorted lists."); ch = d.chain_decisions()
check("coding chain: qwen3 certified, then uncertified fallbacks", [x.model for x in ch][:1] == ["qwen3"] and not ch[1].certified and ch[1].verify_required)
# disabled -> plain router
cfg2 = dict(CFG); cfg2["routing"] = {}
check("no routing block -> original keyword Router", type(make_router(cfg2, MODEL_ROSTER, fm)) is Router)
check("classifier is ASCII-safe for headers", all(ord(c) < 128 for c in sel("Write a function").reason))
print("RESULT %d/%d" % (sum(res), len(res))); sys.exit(0 if all(res) else 1)
