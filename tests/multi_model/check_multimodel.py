
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
def check(name,cond): ok.append((name,bool(cond))); print(("PASS " if cond else "FAIL ")+name)
for m in ("gemma","deepseek","qwen3","fast"):
    with M.use(m) as url: pass
st=M.status()
check("all four resident",set(st["resident"])=={"gemma","deepseek","qwen3","fast"})
check("layout honoured (gemma,deepseek GPU0; qwen3,fast GPU1)",{k:v["gpu"] for k,v in st["resident"].items()}=={"gemma":0,"deepseek":0,"qwen3":1,"fast":1})
check("ports per model", [M._port(x) for x in ("gemma","deepseek","qwen3","fast")]==[8101,8102,8103,8104])
check("final VRAM within card (GPU0 %d, GPU1 %d MiB of 15360)"%(used[0],used[1]), used[0]<15360-1200 and used[1]<15360-1200)
n=len(LOG)
with M.use("gemma"): pass
check("resident model re-used with no reload",len(LOG)==n)
with M.use("quant") as u: check("alias resolves to qwen3 port",u.endswith("8103"))
# parallel: two different models busy at once
res=[]
def hold(m):
    with M.use(m): time.sleep(0.4); res.append(time.time())
t0=time.time(); a=threading.Thread(target=hold,args=("gemma",)); b=threading.Thread(target=hold,args=("qwen3",)); a.start(); b.start(); a.join(); b.join()
check("different models run in parallel (%.2fs for 2x0.4s)"%(time.time()-t0), time.time()-t0<0.7)
# same model serialises (np=1)
t0=time.time(); a=threading.Thread(target=hold,args=("gemma",)); b=threading.Thread(target=hold,args=("gemma",)); a.start(); b.start(); a.join(); b.join()
check("same model requests are queued (%.2fs)"%(time.time()-t0), time.time()-t0>=0.75)
# busy model is never evicted; if nothing fits -> bounded wait -> clean error (never forced)
used[0]+=9000                      # foreign memory on GPU0 (e.g. an unrelated process)
cfg2=dict(cfg,residency=dict(cfg["residency"],layout={"gemma":{"gpu":0,"port":8101},"deepseek":{"gpu":0,"port":8102}}))
M2=R.make_manager(Reg(),cfg2); 
with M2.use("gemma"):               # gemma is BUSY while deepseek is requested
    try:
        used[1]+=14000              # GPU1 also full
        with M2.use("deepseek"): pass
        check("insufficient VRAM raises, never forces",False)
    except (R.LoadError,R.QueueTimeout) as e: check("insufficient VRAM -> clean error after bounded wait: "+type(e).__name__,True)
    check("busy model was not evicted","gemma" in M2.servers)
used[0]-=9000; used[1]-=14000
# unhealthy resident model reloads
M.servers["fast"].alive=False; n=len(LOG)
with M.use("fast"): pass
check("unhealthy model restarted",any(e[0]=="stop" and e[1]=="fast" for e in LOG[n:]) or len(LOG)>n)
# load failure cooldown
FAIL["gemma"]=True; M.servers.pop("gemma").stop() if "gemma" in M.servers else None; M.gpu_of.pop("gemma",None)
try:
    with M.use("gemma"): pass
    check("failed load raises",False)
except R.LoadError as e: check("failed load raises LoadError",True)
t0=time.time()
try:
    with M.use("gemma"): pass
except R.LoadError as e: check("cooldown refuses immediate retry (%.2fs)"%(time.time()-t0), "recently" in str(e))
# a successful load slower than queue_timeout_s must still serve the request
class SlowServer(FakeServer):
    def start(self):
        time.sleep(4); return FakeServer.start(self)
R.LlamaServer=SlowServer
M3=R.make_manager(Reg(),dict(cfg,residency=dict(cfg["residency"],queue_timeout_s=3)))
try:
    with M3.use("qwen3") as u: check("slow load (4s > queue_timeout 3s) still served",True)
except Exception as e: check("slow load still served: "+repr(e)[:60],False)
R.LlamaServer=FakeServer
print("SUMMARY",sum(1 for _,c in ok if c),"/",len(ok))
