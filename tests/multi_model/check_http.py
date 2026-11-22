
import sys, json, threading, time, http.client, contextlib
sys.path.insert(0, sys.argv[1])
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import maple.api as A
from maple.runtime import LoadError
CFG = json.load(open(sys.argv[2])); CFG["host"] = "127.0.0.1"
PORTS = {"gemma": 18101, "deepseek": 18102, "qwen3": 18103, "fast": 18104}
SEEN = {}                                  # model -> last body received by its upstream
class Up(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); m = self.server.mid; SEEN[m] = b
        out = json.dumps({"id": "x", "model": m, "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi from " + m}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
ups = []
for m, p in PORTS.items():
    s = ThreadingHTTPServer(("127.0.0.1", p), Up); s.mid = m; threading.Thread(target=s.serve_forever, daemon=True).start(); ups.append(s)
class Reg:
    def ids(self): return list(PORTS)
    def get(self, x): return {"id": {"quant": "qwen3"}.get(x, x)}
    def list(self): return [{"id": m} for m in PORTS]
    def check_all(self): return [{"id": m, "status": "ok"} for m in PORTS]
    def aliases(self): return {"quant": "qwen3"}
class Mgr:
    def __init__(s): s.fail = set(); s.res = {m: {"gpu": 0 if m in ("gemma", "deepseek") else 1, "busy": False, "waiting": 0} for m in PORTS}; s.failed = {}
    def snapshot(s): return {"resident": {m: v for m, v in s.res.items() if m not in s.failed}, "failed": dict(s.failed), "waiting": {}}
    @contextlib.contextmanager
    def use(s, m, ctx=None):
        m = {"quant": "qwen3"}.get(m, m)
        if m in s.fail:
            s.failed[m] = 30.0; raise LoadError("load failed for " + m)
        yield "http://127.0.0.1:%d" % PORTS[m]
    def status(s): return {"loaded": None, "resident": s.res, "switches": 4, "healthy": True}
    def unload(s): pass
mg = Mgr(); app = A.App(registry=Reg(), manager=mg, cfg=CFG)
httpd = ThreadingHTTPServer(("127.0.0.1", 18000), A.make_handler(app)); httpd.daemon_threads = True
threading.Thread(target=httpd.serve_forever, daemon=True).start(); time.sleep(0.3)
def call(path, body, method="POST"):
    c = http.client.HTTPConnection("127.0.0.1", 18000, timeout=10); c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse(); raw = r.read(); h = {k.lower(): v for k, v in r.getheaders()}; c.close()
    try: return r.status, json.loads(raw), h
    except ValueError: return r.status, raw, h
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)
def chat(txt, model="auto", **extra): return call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": txt}], "max_tokens": 5, **extra})
s, o, h = chat("Write a Python function that merges two sorted lists.")
check("auto coding -> 200 from qwen3 with certified headers", s == 200 and o["model"] == "qwen3" and h["x-maple-certified"] == "true" and h["x-maple-category"] == "coding" and h["x-maple-model"] == "qwen3")
check("body x_maple.qualification present", o["x_maple"]["qualification"]["category"] == "coding" and o["x_maple"]["qualification"]["certified"] is True)
s, o, h = chat("Write unit tests for a function that parses ISO dates.")
check("auto test-generation -> served but uncertified, verify required (%s)" % o["model"], s == 200 and h["x-maple-certified"] == "false" and h["x-maple-verify"] == "required" and o["x_maple"]["qualification"]["tier"] == "best_effort")
s, o, h = chat("Write unit tests for a function that parses ISO dates.", maple={"require_certified": True})
check("require_certified on test generation -> 409 no_qualified_model", s == 409 and "no_qualified_model" in json.dumps(o))
s, o, h = chat("Write a Python function that merges two sorted lists.", maple={"category": "debugging", "exclude": ["qwen3"]})
check("caller category hint + exclude honoured (-> gemma)", s == 200 and o["model"] == "gemma" and o["x_maple"]["qualification"]["category"] == "debugging")
check("'maple' control field not forwarded to llama-server", "maple" not in SEEN["gemma"] and SEEN["gemma"]["model"] == "gemma")
s, o, h = chat("hello", model="deepseek")
check("explicit deepseek -> deepseek, uncertified, not substituted", s == 200 and o["model"] == "deepseek" and h["x-maple-certified"] == "false" and o["x_maple"]["explicit"] is True)
s, o, h = chat("Write a function", model="quant"); check("explicit alias quant -> qwen3", s == 200 and o["model"] == "qwen3")
s, o, h = chat("Write a function", model="nonexistent"); check("unknown model -> 404", s == 404)
# load failure during routing: qwen3 fails for a coding task -> flagged fallback to gemma
mg.fail.add("qwen3")
s, o, h = chat("Write a Python function that merges two sorted lists.")
check("qwen3 load failure -> fallback to gemma, flagged uncertified (%s)" % h.get("x-maple-certified"), s == 200 and o["model"] == "gemma" and h["x-maple-certified"] == "false" and h["x-maple-verify"] == "required" and "fallback" in h["x-maple-route"].lower())
s, o, h = chat("Write a Python function that merges two sorted lists.", model="qwen3")
check("explicit qwen3 with load failure -> controlled 5xx error, NOT silently replaced", s in (502, 503) and "qwen3" in json.dumps(o))
s, o, h = chat("Solve: 3x + 7 = 25. Think step by step.")
check("reasoning with qwen3 down -> 503 no_suitable_model (no unsafe fallback)", s == 503 and "no_suitable_model" in json.dumps(o))
mg.fail.clear(); mg.failed.clear()
s, o, h = call("/v1/route", {"messages": [{"role": "user", "content": "Design a multi-tenant billing service architecture."}]})
check("/v1/route preview includes qualification, runs nothing", s == 200 and o["model"] == "qwen3" and o["qualification"]["category"] == "architecture")
s, o, h = call("/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]})
check("tools present -> api_tool category", s == 200 and h["x-maple-category"] == "api_tool")
s, o, h = call("/v1/completions", {"model": "auto", "prompt": "Write a Python function that sorts a list.", "max_tokens": 5})
check("legacy /v1/completions routes (%s -> %s)" % (s, o.get("model") if isinstance(o, dict) else o), s == 200 and o["model"] in PORTS and "x-maple-category" in h)
s, o, h = call("/health", None, "GET"); check("/health still served", s == 200)
s, o, h = call("/v1/models", None, "GET"); check("/v1/models still served (%s)" % s, s == 200)
print("RESULT %d/%d" % (sum(res), len(res))); sys.exit(0 if all(res) else 1)
