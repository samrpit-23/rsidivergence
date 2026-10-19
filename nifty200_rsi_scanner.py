"""
NIFTY 200 - Bullish RSI Divergence Reversal Scanner
(Daily / Weekly / Monthly, Technical Score + separate Financial Score)

Setup:   pip install streamlit yfinance pandas numpy plotly requests
Run:     streamlit run nifty200_rsi_scanner.py
"""
import io, json, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import time as dtime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.lib.stride_tricks import sliding_window_view as swv

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# left/right = bars each side to confirm a swing low; min/max_gap = distance
# between the two lows; max_recent = how stale the 2nd low may be.
TF = {
    "Daily":   dict(interval="1d",  period="2y",  left=5, right=2, min_gap=5, max_gap=60, max_recent=8, w52=252, long_ema=200, min_bars=120),
    "Weekly":  dict(interval="1wk", period="6y",  left=4, right=2, min_gap=4, max_gap=40, max_recent=4, w52=52,  long_ema=50,  min_bars=80),
    "Monthly": dict(interval="1mo", period="15y", left=3, right=1, min_gap=3, max_gap=24, max_recent=3, w52=12,  long_ema=20,  min_bars=40),
}
NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
]
# Used only if the NSE list can't be downloaded and no CSV is uploaded.
FALLBACK = """RELIANCE HDFCBANK ICICIBANK INFY TCS BHARTIARTL SBIN LT ITC KOTAKBANK AXISBANK HINDUNILVR
BAJFINANCE MARUTI SUNPHARMA ASIANPAINT TITAN ULTRACEMCO NESTLEIND WIPRO HCLTECH ONGC NTPC POWERGRID
TATASTEEL JSWSTEEL M&M BAJAJ-AUTO BAJAJFINSV ADANIENT ADANIPORTS COALINDIA GRASIM TECHM INDUSINDBK
CIPLA DRREDDY EICHERMOT HEROMOTOCO BRITANNIA DIVISLAB APOLLOHOSP HDFCLIFE SBILIFE BPCL HINDALCO
SHRIRAMFIN TATACONSUM TRENT BEL HAL DLF PIDILITIND GODREJCPL DABUR HAVELLS SIEMENS ABB CHOLAFIN
MUTHOOTFIN PFC RECLTD IOC GAIL VEDL JINDALSTEL SAIL BANKBARODA PNB CANBK IDFCFIRSTB FEDERALBNK AUBANK
LICHSGFIN TVSMOTOR ASHOKLEY BOSCHLTD MRF PAGEIND COLPAL MARICO UPL AMBUJACEM SHREECEM LUPIN
TORNTPHARM ZYDUSLIFE AUROPHARMA ALKEM MAXHEALTH INDHOTEL IRCTC INDIGO NAUKRI POLYCAB CUMMINS LODHA
GODREJPROP OBEROIRLTY PERSISTENT COFORGE MPHASIS"""
FIN_CACHE = "fin_cache.json"
FIN_KEYS = ["returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins", "revenueGrowth",
            "earningsGrowth", "debtToEquity", "currentRatio", "trailingPE", "forwardPE", "priceToBook",
            "freeCashflow", "heldPercentInstitutions", "dividendYield", "sector", "marketCap", "shortName"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _fetch_nse():
    """Raises on failure, so a failed attempt is never cached."""
    err = None
    for u in NSE_URLS:
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
            if len(syms) > 100:
                return syms
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"NSE download failed: {err}")


def load_universe():
    """NSE live list -> local nifty200.csv (commit it to your repo) -> built-in fallback."""
    try:
        return _fetch_nse(), "NSE official list"
    except Exception:
        pass
    for f in ("nifty200.csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty200.csv")):
        if os.path.exists(f):
            try:
                return pd.read_csv(f)["Symbol"].str.strip().tolist(), "local nifty200.csv"
            except Exception:
                pass
    return FALLBACK.split(), "built-in fallback list (~100 stocks; NSE blocked or offline)"


def _fetch_chunk(tickers, interval, period):
    out = {}
    try:
        df = yf.download(tickers, interval=interval, period=period, group_by="ticker",
                         auto_adjust=True, threads=True, progress=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    lvl0 = set(df.columns.get_level_values(0))
    for t in tickers:
        if t in lvl0:
            d = df[t].dropna(subset=["Close"])
            if len(d):
                out[t[:-3]] = d
    return out


@st.cache_data(ttl=900, show_spinner=False)
def download(symbols: tuple, interval: str, period: str):
    """Bulk, multi-threaded download. Retries missing tickers once. Raises if nothing came back
    (so a rate-limited empty result is NOT cached)."""
    tickers = [s + ".NS" for s in symbols]
    out = {}
    for i in range(0, len(tickers), 100):
        out.update(_fetch_chunk(tickers[i:i + 100], interval, period))
    missing = [t for t in tickers if t[:-3] not in out]
    if missing and len(missing) < len(tickers):      # partial failure -> retry the gaps once
        time.sleep(2)
        for i in range(0, len(missing), 50):
            out.update(_fetch_chunk(missing[i:i + 50], interval, period))
    if not out:
        raise RuntimeError("No price data returned (Yahoo Finance may be rate-limiting this server). Try again in a minute.")
    return out, [s for s in symbols if s not in out]


def drop_forming(df, tf):
    """Remove the still-forming candle (approximation, IST market close 15:30)."""
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    last = pd.Timestamp(df.index[-1]).tz_localize(None)
    after_close = now.time() >= dtime(15, 45)
    if tf == "Daily":
        forming = last.date() == now.date() and not after_close
    elif tf == "Weekly":
        week_done = now.weekday() >= 5 or (now.weekday() == 4 and after_close)
        forming = (now.normalize() - last.normalize()).days < 7 and not week_done
    else:
        forming = (last.year, last.month) == (now.year, now.month)
    return df.iloc[:-1] if forming and len(df) > 1 else df


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def rsi_w(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


# ----------------------------------------------------------------------------
# Technical analysis: bullish (regular) RSI divergence + reversal confirmation
# ----------------------------------------------------------------------------
def analyze(df, cfg, p):
    if len(df) < cfg["min_bars"]:
        return None
    cs = df["Close"].astype(float)
    c, h, l, o = (df[k].to_numpy(float) for k in ("Close", "High", "Low", "Open"))
    v = df["Volume"].to_numpy(float)
    n, L = len(c), len(c) - 1
    rsi = rsi_w(cs).to_numpy()
    ema9, ema20, ell = (ema(cs, k).to_numpy() for k in (9, 20, cfg["long_ema"]))
    macd = ema(cs, 12) - ema(cs, 26)
    hist = (macd - ema(macd, 9)).to_numpy()
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    if not atr[L] > 0 or np.isnan(rsi[L]):
        return None

    # swing lows (confirmed with `left` bars before and `right` bars after)
    left, right = cfg["left"], cfg["right"]
    w = left + right + 1
    piv = np.where(l[left:n - right] == swv(l, w).min(axis=1))[0] + left
    piv = piv[~np.isnan(rsi[piv])]
    if len(piv) < 2:
        return None

    def is_clean(a, b):  # RSI & price must not cut through the divergence lines
        x = np.arange(a, b + 1)
        rl = np.interp(x, [a, b], [rsi[a], rsi[b]])
        pl = np.interp(x, [a, b], [l[a], l[b]])
        return bool((rsi[a:b + 1] >= rl - 1.5).all() and (c[a:b + 1] >= pl * 0.99).all())

    best = None
    for p2 in piv[-3:][::-1]:
        age = L - p2
        if age > cfg["max_recent"] or l[p2 + 1:].min() < l[p2]:
            continue  # stale or already invalidated by a lower low
        for p1 in piv[piv < p2]:
            gap = p2 - p1
            if not cfg["min_gap"] <= gap <= cfg["max_gap"]:
                continue
            if l[p2] > l[p1] * 1.003:                       # need lower low / double bottom
                continue
            if gap > 1 and l[p1 + 1:p2].min() < l[p2]:      # p2 must be the lowest since p1
                continue
            dr = rsi[p2] - rsi[p1]
            if dr < p["min_rsi_diff"] or rsi[p1] > p["rsi1_max"]:
                continue
            q = dr - 0.5 * age
            if best is None or q > best[0]:
                best = (q, p1, p2, dr)
    if best is None:
        return None
    _, p1, p2, dr = best
    age, gap = L - p2, p2 - p1

    entry = c[L]
    swing_high = h[p1:p2 + 1].max()
    sl = l[p2] - 0.2 * atr[L]
    ext = (entry - l[p2]) / atr[L]
    if ext > p["max_ext"]:
        return None
    risk = entry - sl
    brk = entry > swing_high
    tgt = entry + 2 * risk if brk else swing_high
    rr = (tgt - entry) / risk

    # ---- confirmation features
    rise1 = rsi[L] > rsi[L - 1]
    rise3 = rise1 and rsi[L - 1] > rsi[L - 2]
    above_sma = rsi[L] > pd.Series(rsi).rolling(5).mean().iloc[-1]
    crossed = any(rsi[L - k - 1] < lv <= rsi[L - k] for k in range(3) for lv in (30, 40, 50))
    green = c[L] > o[L]

    def candle(i):
        body, rng = abs(c[i] - o[i]), h[i] - l[i] + 1e-9
        lw = min(c[i], o[i]) - l[i]
        hammer = lw >= 2 * body and lw >= 0.55 * rng and (h[i] - max(c[i], o[i])) <= 0.25 * rng
        engulf = c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]
        return hammer or engulf
    pat = any(candle(i) for i in range(max(p2, L - 2), L + 1))
    vavg = np.nanmean(v[max(0, L - 20):L])
    vr = max(v[L], v[L - 1]) / vavg if vavg and vavg > 0 else 0
    mdiv = hist[p2] > hist[p1]
    clean = is_clean(p1, p2)

    # ---- context
    w52 = cfg["w52"]
    lo52, hi52 = l[-w52:].min(), h[-w52:].max()
    near_ema = n >= cfg["long_ema"] and abs(l[p2] / ell[p2] - 1) <= 0.06
    near_low = l[p2] <= lo52 * 1.05
    dd = 1 - entry / hi52

    # ---- scoring (100): divergence 40 | freshness 10 | confirmation 30 | context 10 | R:R 10
    r1 = rsi[p1]
    ll = (l[p1] - l[p2]) / l[p1] * 100
    A = min(dr, 10) / 10 * 14
    A += 8 if r1 <= 30 else 6 if r1 <= 35 else 4 if r1 <= 40 else 2 if r1 <= 45 else 0
    A += min(ll, 6) if ll > 0.3 else 3
    A += 6 * clean + (2 if gap >= 0.2 * cfg["max_gap"] else 1) + 4 * mdiv
    B = max(2, 10 - 8 * max(0, age - right) / max(1, cfg["max_recent"] - right))
    rsi_pts = (5 if rise3 else 3 if rise1 else 0) + 3 * above_sma + 2 * crossed
    trig = 10 if brk else 4 * (green and entry > h[L - 1]) + 3 * (entry > ema9[L]) + 3 * (entry > ema20[L])
    C = rsi_pts + trig + 5 * pat + (5 if vr >= 1.5 else 3 if vr >= 1.2 else 1 if vr >= 1 else 0)
    D = 4 * near_ema + 3 * near_low + 3 * (dd >= 0.25)
    E = 10 if rr >= 3 else 7 if rr >= 2 else 4 if rr >= 1.5 else 2 if rr >= 1 else 0
    score = int(round(min(100, A + B + C + D + E)))
    grade = "A+" if score >= 75 else "A" if score >= 65 else "B" if score >= 50 else "C"
    stage = "Breakout" if brk else "Triggered" if (green and entry > h[L - 1] and rise1) else "Watch"

    notes = [x for x, ok in [("RSI rising", rise1), ("RSI>avg", above_sma), ("Reversal candle", pat),
                             (f"Vol {vr:.1f}x", vr >= 1.2), ("MACD div", mdiv), ("Clean lines", clean),
                             ("Near long EMA", near_ema), ("Near 52w low", near_low)] if ok]
    idx = df.index
    return {
        "Stage": stage, "Tech Score": score, "Grade": grade,
        "Div Type": "Regular (LL vs HL)" if ll > 0.3 else "Double-bottom",
        "CMP": round(float(entry), 2), "RSI@Low1": round(float(r1), 1), "RSI@Low2": round(float(rsi[p2]), 1),
        "RSI Now": round(float(rsi[L]), 1), "Low1": round(float(l[p1]), 2), "Low2": round(float(l[p2]), 2),
        "Low1 Date": pd.Timestamp(idx[p1]).strftime("%d-%b-%y"),
        "Low2 Date": pd.Timestamp(idx[p2]).strftime("%d-%b-%y"),
        "Bars Ago": int(age), "SL": round(float(sl), 2), "Target": round(float(tgt), 2), "R:R": round(float(rr), 2),
        "Risk %": round(float(risk / entry * 100), 1), "Why": ", ".join(notes),
        "_p1": int(p1), "_p2": int(p2),
    }


# ----------------------------------------------------------------------------
# Financial score
# ----------------------------------------------------------------------------
def num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def fin_score(info):
    parts, tot = {}, [0]
    is_fin = info.get("sector") == "Financial Services"

    def add(name, v, tiers, mx, higher=True):
        tot[0] += mx
        v = num(v)
        if v is None:
            return
        parts[name] = (next((pts for t, pts in tiers if (v >= t if higher else v <= t)), 0), mx)

    # Profitability 25
    add("ROE", info.get("returnOnEquity"), [(.20, 10), (.15, 8), (.10, 5), (0, 2)], 10)
    add("Net margin", info.get("profitMargins"), [(.15, 8), (.10, 6), (.05, 4), (0, 1)], 8)
    add("Op margin", info.get("operatingMargins"), [(.20, 7), (.12, 5), (.06, 3), (0, 1)], 7)
    # Growth 25
    add("Rev growth", info.get("revenueGrowth"), [(.20, 12), (.12, 9), (.06, 6), (0, 3)], 12)
    add("EPS growth", info.get("earningsGrowth"), [(.25, 13), (.15, 10), (.05, 6), (0, 3)], 13)
    # Balance sheet 20 (not meaningful for banks/NBFCs -> ROA instead)
    if is_fin:
        add("ROA", info.get("returnOnAssets"), [(.015, 10), (.01, 7), (.005, 4), (0, 1)], 10)
    else:
        add("D/E", info.get("debtToEquity"), [(30, 12), (80, 9), (150, 5), (250, 2)], 12, higher=False)
        add("Current ratio", info.get("currentRatio"), [(2, 8), (1.5, 6), (1, 3)], 8)
    # Valuation 20
    pe = num(info.get("trailingPE"))
    add("PE", None if pe is None else (pe if pe > 0 else 1e9), [(15, 10), (25, 8), (35, 5), (50, 2)], 10, higher=False)
    add("P/B", info.get("priceToBook"), [(2, 5), (4, 4), (7, 2)], 5, higher=False)
    fpe = num(info.get("forwardPE"))
    add("Fwd PE<PE", None if (fpe is None or pe is None) else (1 if fpe < pe else 0), [(1, 5), (0, 1)], 5)
    # Quality / cash 10
    if not is_fin:
        add("FCF>0", None if num(info.get("freeCashflow")) is None else float(num(info["freeCashflow"]) > 0),
            [(1, 5), (0, 0)], 5)
    add("Inst. holding", info.get("heldPercentInstitutions"), [(.25, 3), (.10, 2), (0, 1)], 3)
    add("Dividend", info.get("dividendYield"), [(0.0001, 2)], 2)

    avail = sum(m for _, m in parts.values())
    if not avail:
        return np.nan, 0.0
    return sum(e for e, _ in parts.values()) / avail * 100, avail / tot[0]


def get_fin(symbols):
    try:
        cache = json.load(open(FIN_CACHE)) if os.path.exists(FIN_CACHE) else {}
    except Exception:
        cache = {}
    now = time.time()
    todo = [s for s in symbols if s not in cache or now - cache[s]["ts"] > 86400]

    def work(s):
        for _ in range(2):
            try:
                info = yf.Ticker(s + ".NS").info
                return s, {k: info.get(k) for k in FIN_KEYS}
            except Exception:
                time.sleep(1)
        return s, None

    with ThreadPoolExecutor(6) as ex:
        for s, r in ex.map(work, todo):
            if r:
                cache[s] = {"ts": now, "d": r}
    try:
        json.dump(cache, open(FIN_CACHE, "w"))
    except Exception:
        pass
    return {s: cache[s]["d"] for s in symbols if s in cache}


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def make_chart(df, r, tf):
    p1, p2 = r["_p1"], r["_p2"]
    st_i = max(0, p1 - 40)
    d, rs = df.iloc[st_i:], rsi_w(df["Close"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.7, .3], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"), 1, 1)
    xs = [df.index[p1], df.index[p2]]
    fig.add_trace(go.Scatter(x=xs, y=[df.Low.iloc[p1], df.Low.iloc[p2]], mode="lines+markers",
                             line=dict(color="red", width=2), name="Price lows"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=rs.iloc[st_i:], name="RSI(14)", line=dict(color="#7e57c2")), 2, 1)
    fig.add_trace(go.Scatter(x=xs, y=[rs.iloc[p1], rs.iloc[p2]], mode="lines+markers",
                             line=dict(color="green", width=2), name="RSI lows"), 2, 1)
    for y, col in ((r["SL"], "red"), (r["Target"], "green")):
        fig.add_hline(y=y, line_dash="dash", line_color=col, row=1, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    if tf == "Daily":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Nifty 200 RSI Divergence Scanner", layout="wide")
    st.title("📈 Nifty 200 - Bullish RSI Divergence Reversal Scanner")

    with st.sidebar:
        tf = st.radio("Timeframe", list(TF), horizontal=True)
        min_score = st.slider("Min technical score", 0, 100, 40)
        stages = st.multiselect("Stage", ["Breakout", "Triggered", "Watch"], ["Breakout", "Triggered", "Watch"])
        st.caption("Breakout = closed above the swing high between the 2 lows; Triggered = bullish bar "
                   "above prior high with RSI rising; Watch = divergence only, no trigger yet.")
        min_diff = st.slider("Min RSI divergence (points)", 1.0, 15.0, 3.0, 0.5)
        rsi1_max = st.slider("Max RSI at first low", 30, 60, 45)
        max_ext = st.slider("Max move above swing low (ATRs)", 1.0, 10.0, 4.0, 0.5)
        closed_only = st.checkbox("Use closed candles only", False)
        use_fin = st.checkbox("Add Financial Score (separate column)", True)
        up = st.file_uploader("Custom universe CSV (needs 'Symbol' column)", type="csv")
        run = st.button("🔍 Scan", type="primary", width="stretch")

    if run:
        t0 = time.perf_counter()
        if up is not None:
            symbols, src = pd.read_csv(up)["Symbol"].str.strip().tolist(), "uploaded CSV"
        else:
            symbols, src = load_universe()
        cfg = TF[tf]
        params = dict(min_rsi_diff=min_diff, rsi1_max=rsi1_max, max_ext=max_ext)
        with st.spinner(f"Downloading {len(symbols)} stocks ({tf})..."):
            try:
                data, missing = download(tuple(symbols), cfg["interval"], cfg["period"])
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
                return
        rows, frames = [], {}
        for s, df in data.items():
            df = drop_forming(df, tf) if closed_only else df
            r = analyze(df, cfg, params)
            if r:
                r["Symbol"] = s
                rows.append(r)
                frames[s] = df
        rows = [r for r in rows if r["Tech Score"] >= min_score and r["Stage"] in stages]
        if use_fin and rows:
            with st.spinner("Fetching fundamentals (cached for 24h)..."):
                fin = get_fin([r["Symbol"] for r in rows])
            for r in rows:
                info = fin.get(r["Symbol"])
                if not info:
                    continue
                sc, cov = fin_score(info)
                r["Fin Score"] = round(sc) if (sc == sc and cov >= 0.35) else np.nan
                r["Fin Grade"] = ("Strong" if sc >= 70 else "Average" if sc >= 50 else "Weak") if r["Fin Score"] == r["Fin Score"] else "n/a"
                r["Fin Data %"] = round(cov * 100)
                r["Sector"] = info.get("sector")
                mc = num(info.get("marketCap"))
                r["MCap (Cr)"] = round(mc / 1e7) if mc else None
                r["ROE %"] = None if num(info.get("returnOnEquity")) is None else round(info["returnOnEquity"] * 100, 1)
                r["PE"] = None if num(info.get("trailingPE")) is None else round(info["trailingPE"], 1)
        st.session_state["res"] = dict(missing=missing, rows=rows, frames=frames, tf=tf, src=src, n=len(data),
                                       total=len(symbols), secs=time.perf_counter() - t0)

    res = st.session_state.get("res")
    if not res:
        st.info("Pick a timeframe in the sidebar and press **Scan**.")
        return
    rows = res["rows"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks scanned", f"{res['n']}/{res['total']}")
    c2.metric("Setups found", len(rows))
    c3.metric("Scan time", f"{res['secs']:.1f}s")
    st.caption(f"Universe: {res['src']}  |  Timeframe: {res['tf']}")
    if res["missing"]:
        with st.expander(f"{len(res['missing'])} symbols returned no data (delisted / renamed on Yahoo / rate-limited)"):
            st.write(", ".join(res["missing"]))
    if not rows:
        st.warning("No setups match the current filters. Loosen the sliders or try another timeframe.")
        return

    df = pd.DataFrame(rows).drop(columns=["_p1", "_p2"])
    if "Fin Score" in df:
        df["Combined"] = (0.7 * df["Tech Score"] + 0.3 * df["Fin Score"].fillna(df["Tech Score"])).round()
    sort_col = st.radio("Sort by", [c for c in ("Tech Score", "Fin Score", "Combined") if c in df], horizontal=True)
    df = df.sort_values(sort_col, ascending=False, na_position="last")
    first = ["Symbol", "Stage", "Tech Score", "Grade", "Fin Score", "Fin Grade", "Combined", "CMP"]
    df = df[[c for c in first if c in df] + [c for c in df if c not in first]]
    prog = lambda t: st.column_config.ProgressColumn(t, min_value=0, max_value=100, format="%d")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        k: prog(k) for k in ("Tech Score", "Fin Score", "Combined") if k in df})
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), f"rsi_div_{res['tf'].lower()}.csv")

    sym = st.selectbox("Chart", df["Symbol"].tolist())
    r = next(x for x in rows if x["Symbol"] == sym)
    st.plotly_chart(make_chart(res["frames"][sym], r, res["tf"]), width="stretch")
    st.caption(f"Why: {r['Why']}")
    st.caption("Educational tool, not investment advice. Always verify the divergence on the chart "
               "and size positions around the stop-loss.")


if __name__ == "__main__":
    main()
