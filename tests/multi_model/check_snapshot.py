
import sys, types, threading, time, importlib, os
sys.path.insert(0, sys.argv[1])
import maple.runtime as R
import maple.resource_manager as rm
GPU=[{"index":0,"mem_total_mib":15360,"mem_used_mib":448,"mem_free_mib":14912},{"index":1,"mem_total_mib":15360,"mem_used_mib":451,"mem_free_mib":14909}]
used={0:448,1:451}
def fake_gpu(cfg=None): return [dict(g,mem_free_mib=15360-used[g["index"]],mem_used_mib=used[g["index"]]) for g in GPU]
rm.gpu_info=fake_gpu
LOG=[]
class FakeServer:
    def __init__(self,m,cfg,placement,ctx,log_dir): self.m=m["id"]; self.g=placement["devices"][0]; self.port=cfg["llama_port"]; self.alive=False
    def start(self):
        if FAIL.get(self.m): raise R.LoadError("boom "+self.m)
        used[self.g]+=NEED[self.m]; self.alive=True; LOG.append(("start",self.m,self.g,self.port)); return {"load_time_s":0.01,"pid":1,"cmd":[]}
    def healthy(self): return self.alive and not DEAD.get(self.m)
    def stop(self):
        if self.alive: used[self.g]-=NEED[self.m]; self.alive=False; LOG.append(("stop",self.m,self.g))
NEED={"fast":1240,"deepseek":5950,"qwen3":5264,"gemma":5164}; FAIL={}; DEAD={}
R.LlamaServer=FakeServer
class Reg:
    def ids(self): return ["deepseek","gemma","qwen3","fast"]
    def get(self,x): return {"id":{"quant":"qwen3"}.get(x,x)}
    def require(self,x): return {"id":x,"model_path":"/dev/null"}
cfg={"host":"127.0.0.1","llama_port":8081,"default_ctx":4096,"models":{},"residency":{"mode":"multi","vram_margin_mib":1200,"queue_timeout_s":3,"fail_cooldown_s":5,
     "vram_mib":NEED,"layout":{"gemma":{"gpu":0,"port":8101},"deepseek":{"gpu":0,"port":8102},"qwen3":{"gpu":1,"port":8103},"fast":{"gpu":1,"port":8104}}}}
M=R.make_manager(Reg(),cfg); assert isinstance(M,R.MultiModelManager)

ok=[]
def check(name,cond): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ")+name)
for m in ("gemma","qwen3"):
    with M.use(m): pass
s=M.snapshot(); check("snapshot lists resident with gpu/busy/waiting", set(s["resident"])=={"gemma","qwen3"} and s["resident"]["gemma"]["gpu"]==0 and s["resident"]["qwen3"]["waiting"]==0)
seen=[]
def hold(m,t=0.5):
    with M.use(m): time.sleep(t)
a=threading.Thread(target=hold,args=("gemma",)); a.start(); time.sleep(0.15)
b=threading.Thread(target=hold,args=("gemma",0.05)); b.start(); time.sleep(0.15)
s=M.snapshot(); check("busy flag and queue length visible (busy=%s waiting=%s)"%(s["resident"]["gemma"]["busy"],s["resident"]["gemma"]["waiting"]), s["resident"]["gemma"]["busy"] and s["resident"]["gemma"]["waiting"]==1)
a.join(); b.join(); s=M.snapshot(); check("queue counter returns to zero", s["resident"]["gemma"]["waiting"]==0 and not s["resident"]["gemma"]["busy"])
# timeout path must also decrement the counter
cfg3=dict(cfg,residency=dict(cfg["residency"],queue_timeout_s=0.3)); M3=R.make_manager(Reg(),cfg3)
with M3.use("gemma"): pass
def hold3(): 
    with M3.use("gemma"): time.sleep(1.0)
t=threading.Thread(target=hold3); t.start(); time.sleep(0.1)
try:
    with M3.use("gemma"): pass
    check("queue timeout raises",False)
except R.QueueTimeout: check("queue timeout raises QueueTimeout",True)
t.join(); check("waiting counter back to 0 after timeout",M3.snapshot()["resident"]["gemma"]["waiting"]==0)
# exception inside the with-body must free the model and the counter
try:
    with M.use("qwen3"): raise ValueError("x")
except ValueError: pass
with M.use("qwen3"): pass
check("exception in body frees the model",M.snapshot()["resident"]["qwen3"]["waiting"]==0 and not M.snapshot()["resident"]["qwen3"]["busy"])
# failed load shows in snapshot
FAIL["fast"]=True
try:
    with M.use("fast"): pass
except R.LoadError: pass
check("failed load appears in snapshot with cooldown",M.snapshot()["failed"].get("fast",0)>0)
print("RESULT %d/%d"%(sum(ok),len(ok))); sys.exit(0 if all(ok) else 1)
