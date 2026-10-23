#!/usr/bin/env python3
"""Probe: reconstruct draft arm D (6.74x) under alternative conventions;
check days-long fractions (22.9% vs 30.1%), log-return unconditional mean
(12.2bp claim), and ddof variants of the full-sample z arm."""
import json, math, urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}

def fetch(symbol):
    out, cur = [], int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp() * 1000) - 1
    while cur < end:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval=1d&startTime={cur}&endTime={end}&limit=1000")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=45) as r:
            b = json.loads(r.read().decode())
        if not b: break
        out.extend(b); cur = b[-1][0] + 1
        if len(b) < 1000: break
    seen = {int(k[0]): k for k in out}
    ks = [seen[t] for t in sorted(seen)]
    return [float(k[1]) for k in ks], [float(k[4]) for k in ks]

def mean(x): return sum(x)/len(x)
def sd_pop(x):
    m = mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def sd_smp(x):
    m = mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1))

def positions(c, M, zwin, thr, mode, sd_fn):
    n = len(c)
    mom = [None]*n
    for t in range(M, n): mom[t] = c[t]/c[t-M]-1.0
    p = [0]*n
    vals = [m for m in mom if m is not None]
    for t in range(n):
        if mom[t] is None: continue
        if mode == "rolling":
            if t < M+zwin-1: continue
            win = mom[t-zwin+1:t+1]
            mu, sd = mean(win), sd_fn(win)
        elif mode == "full":
            mu, sd = mean(vals), sd_fn(vals)
        elif mode == "fullmean_rollsd":     # hybrid guess
            mu = mean(vals)
            if t < M+zwin-1: continue
            sd = sd_fn(mom[t-zwin+1:t+1])
        elif mode == "rollmean_fullsd":     # hybrid guess 2
            if t < M+zwin-1: continue
            mu = mean(mom[t-zwin+1:t+1]); sd = sd_fn(vals)
        z = (mom[t]-mu)/sd if sd else 0.0
        p[t] = 1 if z > thr else 0
    return p

def c_ledger(c, p, cost=0.001):
    n = len(c); net = [0.0]*n
    for u in range(1, n):
        r = c[u]/c[u-1]-1.0
        prev = p[u-2] if u >= 2 else 0
        net[u] = p[u-1]*r - cost*abs(p[u-1]-prev)
    return net

def perf(net):
    w, peak, mdd = 1.0, 1.0, 0.0
    for r in net:
        w *= 1+r; peak = max(peak, w); mdd = max(mdd, 1-w/peak)
    n = len(net); yrs = n/365.25
    mu = mean(net); sd = sd_smp(net)
    return {"x": round(w,2), "cagr": round((w**(1/yrs)-1)*100,1),
            "sharpe": round(mu/sd*math.sqrt(365.25),2), "mdd": round(mdd*100,1)}

o, c = fetch("BTCUSDT")
n = len(c)
res = {"n": n}
# candidate reconstructions of arm D under C ledger
for mode in ("full", "fullmean_rollsd", "rollmean_fullsd"):
    for name, fn in (("pop", sd_pop), ("smp", sd_smp)):
        p = positions(c, 20, 90, 0.5, mode, fn)
        res[f"D_{mode}_{name}"] = perf(c_ledger(c, p))
# rolling baselines for days-long comparison
p_roll_smp = positions(c, 20, 90, 0.5, "rolling", sd_smp)
p_full_pop = positions(c, 20, 90, 0.5, "full", sd_pop)
p_full_smp = positions(c, 20, 90, 0.5, "full", sd_smp)
res["days_long_pct"] = {
    "rolling_smp": round(100*mean(p_roll_smp),1),
    "full_pop": round(100*mean(p_full_pop),1),
    "full_smp": round(100*mean(p_full_smp),1)}
# unconditional intra-bar mean: simple vs log, full vs post-warmup
intra_s = [c[t]/o[t]-1.0 for t in range(n)]
intra_l = [math.log(c[t]/o[t]) for t in range(n)]
res["intra_all_bp"] = {"simple": round(mean(intra_s)*1e4,1),
                       "log": round(mean(intra_l)*1e4,1),
                       "simple_postwarmup": round(mean(intra_s[110:])*1e4,1),
                       "log_postwarmup": round(mean(intra_l[110:])*1e4,1)}
print(json.dumps(res, indent=1))
