#!/usr/bin/env python3
"""Regime filter grid sweep — multiprocessing version."""
import sys, os, time as _t
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd, numpy as np
from multiprocessing import Pool, cpu_count

def load_regime_data(spot_sym):
    from data.storage.db_manager import get_db
    db = get_db()
    bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime"
    )
    for c in ["open","high","low","close","volume"]: bars[c]=bars[c].astype(float)
    bars["date"]=bars["datetime"].str[:10]

    prev_d=None;ct=0;cv=0;va=[]
    for _,r in bars.iterrows():
        if r["date"]!=prev_d: ct=0;cv=0;prev_d=r["date"]
        ct+=(r["high"]+r["low"]+r["close"])/3*r["volume"];cv+=r["volume"]
        va.append(ct/cv if cv>0 else r["close"])
    bars["vwap"]=va

    def calc_adx(h,l,c,p=14):
        n=len(h);adx=np.full(n,np.nan)
        if n<p*3:return adx
        tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
        pd_=np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]),np.maximum(h[1:]-h[:-1],0),0)
        md=np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]),np.maximum(l[:-1]-l[1:],0),0)
        at=np.zeros(n-1);pi=np.zeros(n-1);mi=np.zeros(n-1)
        at[p-1]=np.mean(tr[:p]);pi[p-1]=np.mean(pd_[:p]);mi[p-1]=np.mean(md[:p])
        for i in range(p,n-1):
            at[i]=(at[i-1]*(p-1)+tr[i])/p;pi[i]=(pi[i-1]*(p-1)+pd_[i])/p;mi[i]=(mi[i-1]*(p-1)+md[i])/p
        pir=100*pi/np.where(at>0,at,1);mir=100*mi/np.where(at>0,at,1)
        dx=100*np.abs(pir-mir)/np.where(pir+mir>0,pir+mir,1)
        as_=np.zeros(n-1);st=p*2-1
        if st<n-1:
            as_[st]=np.mean(dx[p:st+1])
            for i in range(st+1,n-1):as_[i]=(as_[i-1]*(p-1)+dx[i])/p
        adx[1:]=as_
        return adx
    bars["adx"]=calc_adx(bars["high"].values,bars["low"].values,bars["close"].values)

    all_regimes = {}
    for adx_t in [15, 20, 25]:
        for vwap_t in [0.002, 0.003, 0.004]:
            dr = {}
            for d, grp in bars.groupby("date"):
                if d<"2025-05-16" or len(grp)<6: continue
                ih=grp["high"].iloc[:6].max();il=grp["low"].iloc[:6].min()
                sc=[]
                for i in range(len(grp)):
                    p=grp.iloc[i]["close"];v=grp.iloc[i]["vwap"];a=grp.iloc[i]["adx"]
                    cA=p>ih or p<il
                    ap=grp.iloc[max(0,i-3)]["adx"]
                    cB=pd.notna(a) and a>adx_t and (pd.notna(ap) and a>ap)
                    vd=abs(p-v)/v if v>0 else 0
                    nc=True
                    if i>=5:
                        for j in range(max(0,i-4),i):
                            if (grp.iloc[j]["close"]-grp.iloc[j]["vwap"])*(grp.iloc[j+1]["close"]-grp.iloc[j+1]["vwap"])<0:
                                nc=False;break
                    cC=vd>vwap_t and nc
                    sc.append(int(cA)+int(cB)+int(cC))
                dr[d.replace("-","")]=sc
            all_regimes[(adx_t, vwap_t)] = dr
    return all_regimes


def run_one_config(args):
    sym, adx_t, vwap_t, thr_label, thr_map, regime_data, dates = args
    import sys; sys.path.insert(0, '/Users/kevinzhao/Documents/options_quant_trading')
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day
    import strategies.intraday.A_share_momentum_signal_v2 as sm

    db = get_db()
    orig = sm.SignalGeneratorV2.score_all

    def _patched(self, symbol, bar_5m, bar_15m, daily_bar, quote_data=None,
                 sentiment=None, zscore=None, is_high_vol=True, d_override=None, vol_profile=None):
        r = orig(self, symbol, bar_5m, bar_15m, daily_bar, quote_data, sentiment, zscore, is_high_vol, d_override, vol_profile)
        if r is None: return r
        if "datetime" in bar_5m.columns:
            dt = str(bar_5m.iloc[-1]["datetime"]); td = dt[:10].replace("-","")
        else: return r
        rs = regime_data.get(td, [])
        if not rs: return r
        today = bar_5m[bar_5m["datetime"].astype(str).str.startswith(dt[:10])]
        idx = min(len(today)-1, len(rs)-1)
        s = rs[idx] if idx >= 0 else 0
        if r["total"] < thr_map.get(s, 60): r["total"] = 0; r["direction"] = ""
        return r

    sm.SignalGeneratorV2.score_all = _patched
    pnl = 0; n = 0
    for td in dates:
        trades = run_day(sym, td, db, verbose=False)
        full = [t for t in trades if not t.get("partial")]
        pnl += sum(t["pnl_pts"] for t in full); n += len(full)
    sm.SignalGeneratorV2.score_all = orig
    return {"sym": sym, "adx": adx_t, "vwap": vwap_t, "thr": thr_label, "pnl": pnl, "n": n}


if __name__ == "__main__":
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day

    db = get_db()
    dates_df = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        "WHERE symbol='000852' AND period=300 AND d >= '2025-05-16' ORDER BY d"
    )
    ALL_DATES = [d.replace('-','') for d in dates_df['d'].tolist()]

    print(f"Regime sweep (parallel) | {len(ALL_DATES)} days | {cpu_count()} cores\n", flush=True)

    # Precompute regimes
    print("Precomputing regimes...", flush=True)
    t0 = _t.time()
    im_regimes = load_regime_data("000852")
    ic_regimes = load_regime_data("000905")
    print(f"  Done ({_t.time()-t0:.0f}s)\n", flush=True)

    # Baselines
    bp_im = 0; bp_ic = 0
    for td in ALL_DATES:
        t = run_day("IM", td, db, verbose=False)
        bp_im += sum(x["pnl_pts"] for x in t if not x.get("partial"))
        t = run_day("IC", td, db, verbose=False)
        bp_ic += sum(x["pnl_pts"] for x in t if not x.get("partial"))
    print(f"  IM Baseline: {bp_im:+.0f}pt\n  IC Baseline: {bp_ic:+.0f}pt\n", flush=True)

    IM_THR = [
        ("60/55/50/45", {0:60,1:55,2:50,3:45}),
        ("65/55/50/45", {0:65,1:55,2:50,3:45}),
        ("60/55/50/40", {0:60,1:55,2:50,3:40}),
        ("60/55/45/40", {0:60,1:55,2:45,3:40}),
        ("65/60/50/45", {0:65,1:60,2:50,3:45}),
    ]
    IC_THR = [
        # Fine grid: conservative end close to baseline(60), aggressive end lower
        ("60/58/55/50", {0:60,1:58,2:55,3:50}),
        ("60/55/55/50", {0:60,1:55,2:55,3:50}),
        ("60/55/50/45", {0:60,1:55,2:50,3:45}),
        ("62/58/55/50", {0:62,1:58,2:55,3:50}),
        ("62/60/55/50", {0:62,1:60,2:55,3:50}),
        ("62/58/55/45", {0:62,1:58,2:55,3:45}),
        ("65/60/55/50", {0:65,1:60,2:55,3:50}),
        ("65/58/55/50", {0:65,1:58,2:55,3:50}),
        ("60/60/55/50", {0:60,1:60,2:55,3:50}),
        ("60/60/60/50", {0:60,1:60,2:60,3:50}),
        ("60/60/60/55", {0:60,1:60,2:60,3:55}),
    ]

    tasks = []
    for adx_t in [15, 20, 25]:
        for vwap_t in [0.002, 0.003, 0.004]:
            # Skip IM (already done), only run IC
            for tl, tm in IC_THR:
                tasks.append(("IC", adx_t, vwap_t, tl, tm, ic_regimes[(adx_t,vwap_t)], ALL_DATES))

    print(f"Running {len(tasks)} configs on {min(cpu_count(),8)} workers...", flush=True)
    t0 = _t.time()
    with Pool(min(cpu_count(), 8)) as pool:
        results = pool.map(run_one_config, tasks)
    elapsed = _t.time() - t0
    print(f"Done: {elapsed:.0f}s ({elapsed/len(results):.1f}s/config)\n", flush=True)

    for sym, baseline in [("IM", bp_im), ("IC", bp_ic)]:
        sr = sorted([r for r in results if r["sym"]==sym], key=lambda x:-x["pnl"])
        print(f"  ── {sym} Top 10 (baseline={baseline:+.0f}pt) ──", flush=True)
        for i, r in enumerate(sr[:10]):
            print(f"    {i+1}. ADX>{r['adx']} VWAP>{r['vwap']*100:.1f}% thr={r['thr']}: "
                  f"{r['pnl']:>+6.0f}pt {r['n']}笔 (Δ{r['pnl']-baseline:+.0f})", flush=True)
        print(f"  {sym} Bottom 3:", flush=True)
        for r in sr[-3:]:
            print(f"    ADX>{r['adx']} VWAP>{r['vwap']*100:.1f}% thr={r['thr']}: "
                  f"{r['pnl']:>+6.0f}pt (Δ{r['pnl']-baseline:+.0f})", flush=True)
        print(flush=True)
