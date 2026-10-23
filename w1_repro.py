#!/usr/bin/env python3
"""Independent re-implementation of the W1 experiment for adversarial review.

Pure stdlib. Fetches Binance spot 1d klines (API first, data.binance.vision
monthly/daily zip fallback), rebuilds the arms, flip statistics, correlation
diagnostics, gap decomposition, subperiod split, parameter grids, buy&hold.
Prints a JSON report to stdout. Nothing is written to the vault.
"""
import io, json, math, sys, time, urllib.request, zipfile
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

def http_get(url, timeout=45, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")

def fetch_api(symbol, start_ms, end_ms):
    out, cur = [], start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval=1d&startTime={cur}&endTime={end_ms}&limit=1000")
        batch = json.loads(http_get(url))
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return out

def fetch_vision(symbol, start_ms, end_ms):
    rows = {}
    def parse_ts(v):
        v = str(v)
        if "-" in v or " " in v:
            return int(datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        return int(v)
    y, m = 2017, 8
    while True:
        dt = datetime(y, m, 1, tzinfo=timezone.utc)
        if int(dt.timestamp() * 1000) > end_ms:
            break
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{symbol}/1d/{symbol}-1d-{y:04d}-{m:02d}.zip")
        try:
            z = http_get(url, binary=True)
            with zipfile.ZipFile(io.BytesIO(z)) as zf:
                txt = zf.read(zf.namelist()[0]).decode()
            for line in txt.splitlines():
                if not line or line.startswith("open_time"):
                    continue
                f = line.split(",")
                ts = parse_ts(f[0])
                rows[ts] = [ts, f[1], f[2], f[3], f[4], f[5]]
        except Exception as e:
            print(f"vision {y}-{m:02d}: {e}", file=sys.stderr)
        m += 1
        if m == 13:
            y, m = y + 1, 1
        time.sleep(0.05)
    # daily zips for the final partial month (Oct 2026)
    for d in range(1, 10):
        url = (f"https://data.binance.vision/data/spot/daily/klines/"
               f"{symbol}/1d/{symbol}-1d-2026-10-{d:02d}.zip")
        try:
            z = http_get(url, binary=True)
            with zipfile.ZipFile(io.BytesIO(z)) as zf:
                txt = zf.read(zf.namelist()[0]).decode()
            for line in txt.splitlines():
                if not line or line.startswith("open_time"):
                    continue
                f = line.split(",")
                ts = parse_ts(f[0])
                rows[ts] = [ts, f[1], f[2], f[3], f[4], f[5]]
        except Exception:
            pass
    return [rows[t] for t in sorted(rows)]

def load(symbol):
    start_ms = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp() * 1000) - 1
    try:
        raw = fetch_api(symbol, start_ms, end_ms)
        src = "api"
    except Exception as e:
        print(f"API failed ({e}); falling back to data.binance.vision", file=sys.stderr)
        raw = fetch_vision(symbol, start_ms, end_ms)
        src = "vision"
    seen = {}
    for k in raw:
        seen[int(k[0])] = k
    ks = [seen[t] for t in sorted(seen) if t <= end_ms]
    o = [float(k[1]) for k in ks]
    c = [float(k[4]) for k in ks]
    dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in seen_keys(ks)]
    return o, c, dates, src

def seen_keys(ks):
    return [int(k[0]) for k in ks]

def mean(x):
    return sum(x) / len(x)

def pstd(x):
    mu = mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))

def sstd(x):
    mu = mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))

def build_positions(c, M, zwin, thr, full_sample=False, sd_fn=pstd):
    n = len(c)
    mom = [None] * n
    for t in range(M, n):
        mom[t] = c[t] / c[t - M] - 1.0
    p = [0] * n
    vals = [m for m in mom if m is not None]
    if full_sample:
        mu, sd = mean(vals), sd_fn(vals)
        for t in range(n):
            if mom[t] is None:
                continue
            z = (mom[t] - mu) / sd if sd > 0 else 0.0
            p[t] = 1 if z > thr else 0
    else:
        for t in range(M + zwin - 1, n):
            win = mom[t - zwin + 1:t + 1]
            mu, sd = mean(win), sd_fn(win)
            z = (mom[t] - mu) / sd if sd > 0 else 0.0
            p[t] = 1 if z > thr else 0
    return p

def net_returns(o, c, p, arm, cost=0.001):
    n = len(o)
    roo = [None] * n
    for t in range(n - 1):
        roo[t] = o[t + 1] / o[t] - 1.0
    rcc = [None] * n
    for t in range(1, n):
        rcc[t] = c[t] / c[t - 1] - 1.0
    net = [0.0] * n
    def dpp(t):
        prev = p[t - 1] if t > 0 else 0
        return cost * abs(p[t] - prev)
    for u in range(n):
        if arm == "A":
            if u >= 1 and roo[u] is not None:
                net[u] = p[u - 1] * roo[u] - dpp(u - 1)
        elif arm == "B":
            if u >= 2 and rcc[u] is not None:
                net[u] = p[u - 2] * rcc[u] - dpp(u - 2)
        elif arm == "C" or arm == "D":
            if u >= 1 and rcc[u] is not None:
                net[u] = p[u - 1] * rcc[u] - dpp(u - 1)
        elif arm == "F":
            if roo[u] is not None:
                net[u] = p[u] * roo[u] - dpp(u)
        elif arm == "G":
            if rcc[u] is not None:
                net[u] = p[u] * rcc[u] - dpp(u)
        elif arm == "BH":
            if rcc[u] is not None:
                net[u] = rcc[u]
    return net

def perf(net):
    w, peak, mdd = 1.0, 1.0, 0.0
    for r in net:
        w *= (1.0 + r)
        peak = max(peak, w)
        mdd = max(mdd, 1.0 - w / peak)
    n = len(net)
    yrs = n / 365.25
    cagr = (w ** (1.0 / yrs) - 1.0) * 100 if w > 0 and yrs > 0 else float("nan")
    mu, sd = mean(net), sstd(net)
    sharpe = mu / sd * math.sqrt(365.25) if sd > 0 else float("nan")
    return {"x": round(w, 2), "cagr_pct": round(cagr, 1),
            "sharpe": round(sharpe, 2), "maxdd_pct": round(mdd * 100, 1)}

def pearson(a, b):
    n = len(a)
    ma, mb = mean(a), mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in a))
    db = math.sqrt(sum((v - mb) ** 2 for v in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")

def analyze(symbol):
    o, c, dates, src = load(symbol)
    n = len(o)
    out = {"symbol": symbol, "source": src, "n_bars": n,
           "first_date": dates[0], "last_date": dates[-1]}
    M, zwin, thr = 20, 90, 0.5
    p = build_positions(c, M, zwin, thr)
    arms = {}
    for a in ["A", "B", "C", "F", "G", "BH"]:
        arms[a] = perf(net_returns(o, c, p, a))
    # D arm: full-sample normalization, C ledger
    pD = build_positions(c, M, zwin, thr, full_sample=True)
    arms["D_fullsample"] = perf(net_returns(o, c, pD, "C"))
    out["arms_main"] = arms
    # sensitivity: sample stddev in rolling z
    p_s = build_positions(c, M, zwin, thr, sd_fn=sstd)
    out["sensitivity_sample_std"] = {"A": perf(net_returns(o, c, p_s, "A")),
                                     "F": perf(net_returns(o, c, p_s, "F"))}
    # flips / trigger-day stats
    flips = [t for t in range(1, n) if p[t] != p[t - 1]]
    entries = [t for t in flips if p[t] == 1]
    exits = [t for t in flips if p[t] == 0]
    intra = [(c[t] / o[t] - 1.0) for t in range(n)]
    out["flips"] = {"total": len(flips),
                    "per_year": round(len(flips) / (n / 365.25), 1),
                    "entries": len(entries), "exits": len(exits)}
    out["intra_bp"] = {
        "all": round(mean(intra) * 1e4, 1),
        "entry_days": round(mean([intra[t] for t in entries]) * 1e4, 1),
        "exit_days": round(mean([intra[t] for t in exits]) * 1e4, 1)}
    # correlations (contemporaneous vs lagged position vs same-bar intra return)
    tt = list(range(1, n))
    out["corr"] = {
        "contemp": round(pearson([p[t] for t in tt], [intra[t] for t in tt]), 3),
        "lagged": round(pearson([p[t - 1] for t in tt], [intra[t] for t in tt]), 3)}
    # gap decomposition A vs F, bar-aligned
    nF = net_returns(o, c, p, "F")
    nA = net_returns(o, c, p, "A")
    gap = [math.log(1 + nF[u]) - math.log(1 + nA[u]) for u in range(n)
           if (1 + nF[u]) > 0 and (1 + nA[u]) > 0]
    total = sum(gap)
    at_flip = sum(g for u, g in
                  [(u, math.log(1 + nF[u]) - math.log(1 + nA[u])) for u in range(n)
                   if (1 + nF[u]) > 0 and (1 + nA[u]) > 0]
                  if (p[u] != p[u - 1] if u >= 1 else False) or
                     (u + 1 < n and p[u + 1] != p[u]) or
                     (u >= 2 and p[u - 1] != p[u - 2]))
    out["gap"] = {"ln_total": round(total, 4),
                  "implied_inflation_x": round(math.exp(total), 1),
                  "share_at_flips": round(at_flip / total, 4) if total else None,
                  "mean_ln_per_flip": round(total / len(flips), 4) if flips else None}
    # subperiod split at 2022-03-12
    try:
        si = dates.index("2022-03-12")
    except ValueError:
        si = min(range(n), key=lambda i: abs(datetime.strptime(dates[i], "%Y-%m-%d").timestamp() -
                 datetime(2022, 3, 12, tzinfo=timezone.utc).timestamp()))
    def half_ratio(lo, hi):
        wf = math.prod(1 + nF[u] for u in range(lo, hi))
        wa = math.prod(1 + nA[u] for u in range(lo, hi))
        return round(wf / wa, 1) if wa else None
    out["subperiod_F_over_A"] = {"first_half": half_ratio(0, si),
                                 "second_half": half_ratio(si, n),
                                 "split_date": dates[si]}
    # parameter grid (A vs F)
    grid = {}
    for Mg in (10, 20, 60):
        for tg in (0.0, 0.5):
            pg = build_positions(c, Mg, zwin, tg)
            xa = perf(net_returns(o, c, pg, "A"))["x"]
            xf = perf(net_returns(o, c, pg, "F"))["x"]
            grid[f"M{Mg}_T{tg}"] = {"A": xa, "F": xf,
                                    "infl": round(xf / xa, 1) if xa else None}
    out["grid"] = grid
    return out

report = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    try:
        report[sym] = analyze(sym)
    except Exception as e:
        report[sym] = {"error": repr(e)}
print(json.dumps(report, indent=1))
