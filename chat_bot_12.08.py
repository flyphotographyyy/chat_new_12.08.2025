# -*- coding: utf-8 -*-
# Enhanced Stock Signals PRO – Multi-Source Analysis (Streamlit-ready)

import math, re, json, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os, time

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import pytz
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional extras (safe fallbacks)
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Optional price fallback (Stooq via pandas_datareader)
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

APP_TITLE = "📈 Stock Signals PRO – Enhanced Multi-Source Analysis"

# Persistent files (saved in the app's home)
WATCHLIST_FILE = Path.home() / "stock_signals_watchlist.json"
SETTINGS_FILE  = Path.home() / "stock_signals_settings.json"
WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# -------------------- Markets --------------------
MARKETS = {
    "US – NYSE/Nasdaq (09:30–16:00 ET)": {"tz": "America/New_York", "open": (9,30),  "close": (16,0),  "cal": "XNYS"},
    "Germany – XETRA (09:00–17:30 DE)":  {"tz": "Europe/Berlin",    "open": (9,0),   "close": (17,30), "cal": "XETR"},
    "UK – LSE (08:00–16:30 UK)":         {"tz": "Europe/London",    "open": (8,0),   "close": (16,30), "cal": "XLON"},
    "France – Euronext Paris (09:00–17:30 FR)": {"tz": "Europe/Paris", "open": (9,0), "close": (17,30), "cal": "XPAR"},
    "Japan – TSE (09:00–15:00 JST)":     {"tz": "Asia/Tokyo",       "open": (9,0),   "close": (15,0),  "cal": "XTKS"},
    "Australia – ASX (10:00–16:00 AEST)":{"tz": "Australia/Sydney", "open": (10,0),  "close": (16,0),  "cal": "XASX"},
}

NEWS_SOURCES = {
    "Google Finance": "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
    "Yahoo Finance":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US",
}

INDICATOR_CONFIGS = {
    "RSI": {"periods": [14, 21, 30], "overbought": 70, "oversold": 30},
    "MACD": {"fast": [12, 9], "slow": [26, 21], "signal": [9, 7]},
    "Bollinger": {"period": 20, "std_dev": 2},
    "Stochastic": {"k_period": 14, "d_period": 3},
    "Williams_R": {"period": 14},
    "CCI": {"period": 20},
    "MFI": {"period": 14},
}

# -------------------- Utils & persistence --------------------
def load_watchlist() -> List[str]:
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out = []
                for t in data:
                    if isinstance(t, str) and 1 <= len(t.strip()) <= 10:
                        out.append(t.strip().upper())
                return sorted(set(out))
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
    default_watchlist = ["AAPL","MSFT","GOOGL","TSLA","NVDA","AMZN","META"]
    save_watchlist(default_watchlist)
    return default_watchlist

def save_watchlist(tickers: List[str]) -> bool:
    try:
        clean = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        uniq  = sorted(set([t for t in clean if 1 <= len(t) <= 10]))
        WATCHLIST_FILE.write_text(json.dumps(uniq, indent=2, ensure_ascii=False), encoding="utf-8")
        st.sidebar.success(f"✅ Watchlist saved ({len(uniq)})")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Error saving watchlist: {e}")
        return False

def load_settings() -> Dict:
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    default = {
        "risk_profile": "balanced",
        "news_sources": ["Google Finance", "Yahoo Finance"],
        "indicators":   ["RSI","MACD","Bollinger"],
        "lookback_days": 120,
        "news_days": 7,
        "show_charts": True,
        "auto_refresh": True,
    }
    save_settings(default)
    return default

def save_settings(settings: Dict) -> bool:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def now_tz(tz_name: str) -> dt.datetime:
    return dt.datetime.now(pytz.timezone(tz_name))

# -------------------- Caching layers --------------------
@st.cache_data(ttl=900, show_spinner=False)  # 15 minutes
def fetch_price_history(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    """Primary: yfinance; Fallback: Stooq (daily only)."""
    try:
        stock = yf.Ticker(ticker)
        if interval == "30m":
            df = stock.history(period="60d", interval="30m")
        else:
            df = stock.history(period=f"{days}d", interval="1d")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Fallback to Stooq (daily only)
    if interval == "1d" and pdr is not None:
        try:
            start = dt.date.today() - dt.timedelta(days=days + 30)
            end   = dt.date.today()
            d = pdr.DataReader(ticker, "stooq", start=start, end=end)
            if d is not None and not d.empty:
                d = d.sort_index()
                # Ensure columns exist
                for c in ["Open","High","Low","Close","Volume"]:
                    if c not in d.columns:
                        d[c] = np.nan
                return d[["Open","High","Low","Close","Volume"]]
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fast_info(ticker: str) -> Dict:
    out = {}
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        out = {
            "last_price": fi.get("last_price"),
            "market_cap": fi.get("market_cap"),
            "beta": fi.get("beta"),
        }
    except Exception:
        pass
    return out

@st.cache_data(ttl=86400, show_spinner=False)  # 24h fundamentals
def fetch_fundamentals(ticker: str) -> Dict:
    """Best-effort fundamentals from yfinance.info/get_info (may be partial)."""
    info = {}
    try:
        t = yf.Ticker(ticker)
        try:
            info_dict = t.get_info()
        except Exception:
            info_dict = getattr(t, "info", {}) or {}
        if info_dict:
            pe = info_dict.get("trailingPE") or info_dict.get("trailing_pe") or info_dict.get("peRatio")
            fpe = info_dict.get("forwardPE") or info_dict.get("forward_pe")
            div = info_dict.get("dividendYield") or info_dict.get("trailingAnnualDividendYield") or info_dict.get("yield")
            sector = info_dict.get("sector")
            industry = info_dict.get("industry")
            # Normalize dividend to percent if needed
            if isinstance(div, (int,float)) and div is not None and div < 1:
                div = div * 100.0
            info = {
                "trailing_pe": pe,
                "forward_pe": fpe,
                "dividend_yield": div,
                "sector": sector,
                "industry": industry,
            }
    except Exception:
        pass
    return info

@st.cache_data(ttl=900, show_spinner=False)
def fetch_earnings_dates(ticker: str, limit: int = 6):
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        return ed
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)  # 10 minutes
def fetch_news_items(ticker: str, days: int = 7) -> List[dict]:
    items = []
    try:
        google_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:{days}d&hl=en-US&gl=US&ceid=US:en"
        g = feedparser.parse(google_url)
        for e in g.entries[:25]:
            try:
                pub = dt.datetime(*e.published_parsed[:6]) if getattr(e, "published_parsed", None) else dt.datetime.utcnow()
                items.append({"title": e.title, "source": "Google", "published": pub, "url": e.get("link","")})
            except Exception:
                continue
    except Exception:
        pass
    try:
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        y = feedparser.parse(yahoo_url)
        for e in y.entries[:20]:
            try:
                pub = dt.datetime(*e.published_parsed[:6]) if getattr(e, "published_parsed", None) else dt.datetime.utcnow()
                items.append({"title": e.title, "source": "Yahoo", "published": pub, "url": e.get("link","")})
            except Exception:
                continue
    except Exception:
        pass
    return items

@st.cache_data(ttl=60, show_spinner=False)
def cached_is_market_open(market_key: str) -> bool:
    return is_market_open_raw(market_key)

# -------------------- Indicators --------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up  = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal    = ema(macd_line, sig)
    hist      = macd_line - signal
    return macd_line, signal, hist


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Welles Wilder ATR via EMA of True Range."""
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    sd  = series.rolling(period).std()
    upper = mid + std_dev * sd
    lower = mid - std_dev * sd
    return upper, mid, lower


def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    denom = (hh - ll).replace(0, np.nan)
    k = 100 * (close - ll) / denom
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    denom = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / denom


def commodity_channel_index(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    md  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    denom = (0.015 * md).replace(0, np.nan)
    return (tp - sma) / denom


def money_flow_index(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = pd.Series(0.0, index=close.index)
    neg = pd.Series(0.0, index=close.index)
    chg = tp.diff()
    pos[chg > 0] = mf[chg > 0]
    neg[chg < 0] = mf[chg < 0]
    pmf = pos.rolling(period).sum()
    nmf = neg.rolling(period).sum().replace(0, np.nan)
    mr = pmf / nmf
    return 100 - (100 / (1 + mr))


def compute_enhanced_indicators(df: pd.DataFrame, indicator_cfgs: Dict, active_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute indicators based on selected list; defaults to all available."""
    if df.empty: 
        return df
    close = df['Close'].astype(float)
    high  = df['High'].astype(float) if 'High' in df else close
    low   = df['Low'].astype(float)  if 'Low'  in df else close
    volume= df['Volume'] if 'Volume' in df else pd.Series(1_000_000, index=df.index)

    active = set(active_indicators or list(indicator_cfgs.keys()))

    for p in [10,20,50,100,200]:
        df[f'SMA{p}'] = close.rolling(p).mean()
        df[f'EMA{p}'] = ema(close, p)

    # RSI
    if "RSI" in indicator_cfgs:
        for p in indicator_cfgs.get("RSI",{}).get("periods",[14]):
            df[f'RSI{p}'] = rsi(close, p)

    # MACD
    if "MACD" in indicator_cfgs:
        fasts  = indicator_cfgs.get("MACD",{}).get("fast",[12])
        slows  = indicator_cfgs.get("MACD",{}).get("slow",[26])
        sigs   = indicator_cfgs.get("MACD",{}).get("signal",[9])
        for i,(f,s,g) in enumerate(zip(fasts,slows,sigs)):
            suffix = f"_{i+1}" if i>0 else ""
            m, sline, h = macd(close, f, s, g)
            df[f"MACD{suffix}"]     = m
            df[f"MACD_SIG{suffix}"] = sline
            df[f"MACD_HIST{suffix}"] = h

    # Bollinger
    if "Bollinger" in active and "Bollinger" in indicator_cfgs:
        bbp = indicator_cfgs.get("Bollinger",{}).get("period",20)
        bbs = indicator_cfgs.get("Bollinger",{}).get("std_dev",2)
        up, mid, lo = bollinger_bands(close, bbp, bbs)
        df["BB_Upper"]=up; df["BB_Middle"]=mid; df["BB_Lower"]=lo
        width = (up - lo).replace([0, np.inf, -np.inf], np.nan)
        df["BB_Width"] = (width / mid).replace([np.inf, -np.inf], np.nan) * 100
        df["BB_Position"] = np.clip(((close - lo) / width) * 100, 0, 100)

    if "Stochastic" in active:
        k,d = stochastic_oscillator(high, low, close)
        df["Stoch_K"]=k; df["Stoch_D"]=d

    if "Williams_R" in active:
        df["Williams_R"] = williams_r(high, low, close)

    if "CCI" in active:
        df["CCI"] = commodity_channel_index(high, low, close)

    if "MFI" in active:
        df["MFI"] = money_flow_index(high, low, close, volume)

    df["Volume_SMA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = (volume / df["Volume_SMA20"]).replace([np.inf, -np.inf], np.nan)

    # True ATR
    df["ATR"] = atr(high, low, close, 14)

    df["Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    for p in [1,5,10,20]:
        df[f"Return_{p}d"] = close.pct_change(p) * 100

    df["Resistance"] = high.rolling(20).max()
    df["Support"]    = low.rolling(20).min()

    window = min(len(df), 252)
    df["HI52"] = close.rolling(window).max()
    df["LO52"] = close.rolling(window).min()

    return df

# -------------------- Sentiment --------------------
def _clean_title(t: str) -> str:
    return re.sub(r"[\W_]+", " ", (t or "").lower()).strip()


def analyze_sentiment_enhanced(news_items: List[dict]) -> Dict[str, float]:
    if not news_items:
        return {"compound":0.0,"pos":0.0,"neu":1.0,"neg":0.0,"n":0,"confidence":0.0,"recent_trend":0.0}
    vader = SentimentIntensityAnalyzer()
    now = dt.datetime.utcnow()
    seen = set(); scores=[]; weights=[]
    for it in news_items:
        key = _clean_title(it.get("title",""))
        if not key or key in seen: continue
        seen.add(key)
        age_days = max(0.2, (now - it.get("published", now)).total_seconds()/86400.0)
        w = float(np.exp(-age_days/3.0))  # recency weight
        s = vader.polarity_scores(it["title"])['compound']
        if TextBlob:
            try:
                s = 0.7*s + 0.3*TextBlob(it["title"]).sentiment.polarity
            except Exception:
                pass
        scores.append(s); weights.append(w)
    if not scores:
        return {"compound":0.0,"pos":0.0,"neu":1.0,"neg":0.0,"n":0,"confidence":0.0,"recent_trend":0.0}
    wmean = float(np.average(scores, weights=weights))
    std = float(np.std(scores)) if len(scores)>1 else 0.0
    conf = max(0.0, 1.0 - std)  # higher variance => lower confidence
    pos = sum(1 for s in scores if s>0.1)/len(scores)
    neg = sum(1 for s in scores if s<-0.1)/len(scores)
    neu = 1.0 - pos - neg
    recent = float(np.mean(scores[-5:])) if len(scores)>=5 else wmean
    return {"compound":wmean,"pos":pos,"neu":neu,"neg":neg,"n":len(scores),"confidence":conf,"recent_trend":recent}

# -------------------- Market status --------------------
def is_market_open_raw(profile_key: str) -> bool:
    prof = MARKETS.get(profile_key)
    if not prof: return False
    tz = pytz.timezone(prof["tz"])
    now = dt.datetime.now(tz)

    if mcal is not None and prof.get("cal"):
        try:
            cal = mcal.get_calendar(prof["cal"])
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty: return False
            o = sched.iloc[0]["market_open"].tz_convert(tz)
            c = sched.iloc[0]["market_close"].tz_convert(tz)
            return o <= now < c
        except Exception:
            pass
    if now.weekday() > 4:  # weekend
        return False
    (oh,om),(ch,cm) = prof["open"], prof["close"]
    o = now.replace(hour=oh, minute=om, second=0, microsecond=0)
    c = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
    return o <= now < c

# -------------------- Extra analytics --------------------
def earnings_in_days(ticker: str, horizon: int = 14) -> Optional[int]:
    ed = fetch_earnings_dates(ticker, limit=6)
    if ed is None or len(ed)==0: return None
    try:
        idx = ed.index.tz_localize(None)
    except Exception:
        idx = ed.index
    now = dt.datetime.utcnow()
    diffs = [(d.to_pydatetime() - now).days for d in idx if (d.to_pydatetime() - now).days >= 0]
    if not diffs: return None
    dmin = min(diffs)
    return dmin if dmin <= horizon else None

@st.cache_data(ttl=900, show_spinner=False)
def relative_strength_20d(df_asset: pd.DataFrame, bench: str = "SPY") -> pd.Series:
    if df_asset is None or df_asset.empty: return pd.Series(dtype=float)
    b = fetch_price_history(bench, days=len(df_asset)+30, interval="1d")
    if b is None or b.empty: return pd.Series(dtype=float)
    a = df_asset["Close"]
    b = b["Close"].reindex(a.index).ffill()
    rs = (a / b)
    return (rs.pct_change(20).rolling(5).mean()*100).rename("RS_20d_vs_SPY")

def finite(x) -> bool:
    return x is not None and np.isfinite(x)

# -------------------- Classification --------------------
def enhanced_signal_classification(ticker: str, df: pd.DataFrame, news_sent: Dict,
                                   risk_profile: str = "balanced", low_liquidity_cap: float = 1e9) -> Dict:
    if df.empty: return {"error":"No data"}
    cur = df.iloc[-1]
    prev = df.iloc[-2] if len(df)>=2 else cur

    signals = {"technical":0, "momentum":0, "volume":0, "sentiment":0, "fundamental":0}
    reasons = []; conf_factors=[]

    price = float(cur["Close"])
    sma20, sma50, sma200 = cur.get("SMA20",np.nan), cur.get("SMA50",np.nan), cur.get("SMA200",np.nan)
    rsi14, rsi_prev = cur.get("RSI14",np.nan), prev.get("RSI14",np.nan)
    macd_, macds_, macd_prev, macds_prev = cur.get("MACD",np.nan), cur.get("MACD_SIG",np.nan), prev.get("MACD",np.nan), prev.get("MACD_SIG",np.nan)
    bb_pos = cur.get("BB_Position", np.nan)
    vol_ratio = cur.get("Volume_Ratio", np.nan)
    ret5, ret20 = cur.get("Return_5d",np.nan), cur.get("Return_20d",np.nan)
    vol = float(cur.get("Volatility", 0) or 0)

    # Trend via MAs
    if all(finite(x) for x in [sma20, sma50, sma200]):
        if price > sma20 > sma50 > sma200:
            signals["technical"] += 20; reasons.append("Strong uptrend – price > SMA20>SMA50>SMA200"); conf_factors.append(0.9)
        elif price < sma20 < sma50 < sma200:
            signals["technical"] -= 20; reasons.append("Strong downtrend – price < SMA20<SMA50<SMA200"); conf_factors.append(0.9)
        elif price > sma50:
            signals["technical"] += 8; reasons.append("Above medium-term trend"); conf_factors.append(0.6)

    # RSI logic
    if finite(rsi14) and finite(rsi_prev):
        if rsi14 < 30:  signals["technical"] += 12; reasons.append(f"RSI oversold ({rsi14:.1f})");  conf_factors.append(0.8)
        if rsi14 > 70:  signals["technical"] -= 12; reasons.append(f"RSI overbought ({rsi14:.1f})"); conf_factors.append(0.8)
        if rsi_prev < 50 <= rsi14: signals["technical"] += 6; reasons.append("RSI crossed above 50"); conf_factors.append(0.6)
        if rsi_prev > 50 >= rsi14: signals["technical"] -= 6; reasons.append("RSI crossed below 50"); conf_factors.append(0.6)

    # MACD cross
    if all(finite(x) for x in [macd_, macds_, macd_prev, macds_prev]):
        if macd_prev < macds_prev and macd_ > macds_:
            signals["momentum"] += 10; reasons.append("MACD bullish crossover"); conf_factors.append(0.7)
        if macd_prev > macds_prev and macd_ < macds_:
            signals["momentum"] -= 10; reasons.append("MACD bearish crossover"); conf_factors.append(0.7)

    # Bollinger position
    if finite(bb_pos):
        if bb_pos < 10:  signals["technical"] += 8;  reasons.append("Near Bollinger lower band"); conf_factors.append(0.6)
        if bb_pos > 90:  signals["technical"] -= 8;  reasons.append("Near Bollinger upper band"); conf_factors.append(0.6)

    # Volume quality
    if finite(vol_ratio):
        if vol_ratio > 1.5: signals["volume"] += 6; reasons.append(f"High volume ({vol_ratio:.1f}× avg)"); conf_factors.append(0.5)
        if vol_ratio < 0.5: signals["volume"] -= 4; reasons.append("Low volume"); conf_factors.append(0.3)

    # Momentum
    if finite(ret5) and finite(ret20):
        if ret5 > 5 and ret20 > 10:
            signals["momentum"] += 12; reasons.append("Strong positive momentum (5d & 20d)"); conf_factors.append(0.7)
        if ret5 < -5 and ret20 < -10:
            signals["momentum"] -= 12; reasons.append("Strong negative momentum (5d & 20d)"); conf_factors.append(0.7)

    # Relative strength vs SPY
    if "RS_20d_vs_SPY" in df.columns:
        rs = float(cur.get("RS_20d_vs_SPY", 0) or 0)
        if rs > 2:
            signals["momentum"] += 8; reasons.append("Outperforming SPY (20d RS)"); conf_factors.append(0.6)
        elif rs < -2:
            signals["momentum"] -= 8; reasons.append("Underperforming SPY (20d RS)"); conf_factors.append(0.6)

    # Sentiment (recency-weighted)
    if news_sent and news_sent.get("n",0) > 0:
        s = float(news_sent.get("compound",0) or 0)
        c = float(news_sent.get("confidence",0.5) or 0.5)
        if s > 0.3:  signals["sentiment"] += int(10*c); reasons.append(f"Very positive news ({s:+.2f})"); conf_factors.append(min(1.0, c+0.1))
        elif s > 0.1: signals["sentiment"] += int(5*c);  reasons.append(f"Positive news ({s:+.2f})");       conf_factors.append(0.6*c)
        elif s < -0.3:signals["sentiment"] -= int(10*c); reasons.append(f"Very negative news ({s:+.2f})"); conf_factors.append(min(1.0, c+0.1))
        elif s < -0.1:signals["sentiment"] -= int(5*c);  reasons.append(f"Negative news ({s:+.2f})");       conf_factors.append(0.6*c)

    # Fundamentals (from df columns populated by fetch_fundamentals)
    pe_ratio = cur.get("PE_Ratio", np.nan)
    if finite(pe_ratio):
        if pe_ratio < 15: signals["fundamental"] += 6; reasons.append(f"Low P/E ({pe_ratio:.1f})"); conf_factors.append(0.6)
        if pe_ratio > 30: signals["fundamental"] -= 4; reasons.append(f"High P/E ({pe_ratio:.1f})"); conf_factors.append(0.4)

    # Low-liquidity / micro-cap soft penalty
    mcap = cur.get("MarketCap", np.nan)
    if finite(mcap) and mcap < low_liquidity_cap:
        reasons.append("Low market cap – higher noise")
        signals["volume"] -= 4
        conf_factors.append(0.4)

    # Earnings awareness (inside 7 days)
    er_days = earnings_in_days(ticker, 14)
    if er_days is not None and er_days <= 7:
        reasons.append(f"Earnings in {er_days}d – risk elevated")
        signals["technical"] -= 5
        conf_factors.append(0.5)

    # Raw score and normalization to 0–100
    raw = sum(signals.values())
    raw -= min(10, vol/5.0)  # penalize very high volatility a bit
    score = int(np.interp(raw, [-40, 40], [0, 100]))
    score = max(0, min(100, score))

    # Risk thresholds
    thr_buy = {"conservative":65,"balanced":60,"aggressive":55}[risk_profile]
    thr_sell= {"conservative":35,"balanced":40,"aggressive":45}[risk_profile]

    if score >= thr_buy:   signal = "BUY"
    elif score <= thr_sell:signal = "SELL"
    else:                  signal = "HOLD"

    avg_conf = int(min(100, max(0, (np.mean(conf_factors) if conf_factors else 0.5)*100)))

    # Fundamentals enrich (from fast_info if DataFrame lacks)
    fi = fetch_fast_info(ticker)
    beta = fi.get("beta")
    if (cur.get("Beta", np.nan) is np.nan) and beta is not None:
        beta = float(beta)
    else:
        beta = float(cur.get("Beta", beta) or np.nan)

    return {
        "ticker": ticker,
        "signal": signal,
        "score": score,
        "confidence": avg_conf,
        "price": price,
        "signals_breakdown": signals,
        "reasons": reasons[:8],
        "sentiment": news_sent,
        "risk_profile": risk_profile,
        "fundamental_data": {
            "pe_ratio": float(pe_ratio) if finite(pe_ratio) else None,
            "beta": beta if finite(beta) else None,
            "market_cap": int(mcap) if finite(mcap) else None,
            "dividend_yield": float(cur.get("DividendYield", np.nan)) if finite(cur.get("DividendYield", np.nan)) else None,
            "sector": cur.get("Sector", "Unknown"),
            "industry": cur.get("Industry", "Unknown"),
        },
        "earnings_in_days": er_days
    }

# -------------------- Charting --------------------
def add_earnings_vlines(fig, ticker: str, row: int, col: int):
    ed = fetch_earnings_dates(ticker, limit=6)
    if ed is None or len(ed)==0: return
    try:
        idx = ed.index.tz_localize(None)
    except Exception:
        idx = ed.index
    for d in idx:
        fig.add_vline(x=d.to_pydatetime(), line_width=1, line_dash="dot", line_color="gray", row=row, col=col)


def create_enhanced_visualizations(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxis=True, vertical_spacing=0.08, row_heights=[0.6,0.2,0.2],
                        subplot_titles=[f'{ticker} Price', 'RSI', 'MACD'])
    if all(c in df.columns for c in ["Open","High","Low","Close"]):
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                     name=f"{ticker}"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{ticker}"), row=1, col=1)

    for p in [20,50]:
        c = f"SMA{p}"
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c, line=dict(width=1), opacity=0.7), row=1, col=1)

    if all(x in df.columns for x in ["BB_Upper","BB_Lower"]):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="BB Upper", line=dict(width=1, dash="dash"), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="Bollinger", line=dict(width=1, dash="dash"), opacity=0.5, fill="tonexty"), row=1, col=1)

    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    if all(x in df.columns for x in ["MACD","MACD_SIG","MACD_HIST"]):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], mode="lines", name="Signal"), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram", opacity=0.6), row=3, col=1)

    add_earnings_vlines(fig, ticker, 1, 1)

    fig.update_layout(title=f"{ticker} – Enhanced Technicals", xaxis_rangeslider_visible=False, height=800, showlegend=True, template="plotly_white")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI",   row=2, col=1, range=[0,100])
    fig.update_yaxes(title_text="MACD",  row=3, col=1)
    return fig

# -------------------- Backtest (simple demo) --------------------
def simple_backtest(df: pd.DataFrame, hold_days: int = 10) -> Dict:
    if df is None or df.empty or "RSI14" not in df.columns or "MACD" not in df.columns or "MACD_SIG" not in df.columns:
        return {"buy_count":0,"buy_avg":0.0,"sell_count":0,"sell_avg":0.0}
    fwd = df["Close"].pct_change(hold_days).shift(-hold_days)
    buys  = df[(df["RSI14"] < 30) & (df["MACD"] > df["MACD_SIG"])]
    sells = df[(df["RSI14"] > 70) & (df["MACD"] < df["MACD_SIG"])]
    return {
        "buy_count": int(buys.shape[0]),
        "buy_avg": float(fwd.loc[buys.index].mean()*100),
        "sell_count": int(sells.shape[0]),
        "sell_avg": float(fwd.loc[sells.index].mean()*100),
    }

# -------------------- Scanning (parallel) --------------------
def process_one(ticker: str, config: Dict):
    days = config.get("lookback_days", 120)
    interval = config.get("interval","1d")
    use_news = config.get("use_news", True)
    risk = config.get("risk_profile","balanced")

    df = fetch_price_history(ticker, days, interval)
    if df.empty:
        return None, None

    # Fundamentals (fast info + slower fundamentals; written to all rows for easy access)
    fi = fetch_fast_info(ticker)
    for k,v in [("MarketCap","market_cap"),("Beta","beta")]:
        try:
            df[k] = fi.get(v, np.nan)
        except Exception:
            df[k] = np.nan

    fnd = fetch_fundamentals(ticker)
    try:
        df["PE_Ratio"]      = fnd.get("trailing_pe", np.nan)
        df["DividendYield"] = fnd.get("dividend_yield", np.nan)
        df["Sector"]        = fnd.get("sector", "Unknown")
        df["Industry"]      = fnd.get("industry", "Unknown")
    except Exception:
        pass

    # Indicators (use selected list from config)
    df = compute_enhanced_indicators(df, INDICATOR_CONFIGS, config.get("indicators"))

    # relative strength vs SPY
    try:
        df["RS_20d_vs_SPY"] = relative_strength_20d(df, "SPY")
    except Exception:
        df["RS_20d_vs_SPY"] = np.nan

    news_sent = analyze_sentiment_enhanced(fetch_news_items(ticker, config.get("news_days",7))) if use_news else {}
    analysis  = enhanced_signal_classification(ticker, df, news_sent, risk_profile=risk)

    cur = df.iloc[-1]
    row = {
        "Ticker": ticker,
        "Signal": analysis["signal"],
        "Score": analysis["score"],
        "Confidence": f'{analysis["confidence"]}%',
        "Price": f'${analysis["price"]:.2f}',
        "RSI": f'{cur.get("RSI14", np.nan):.1f}' if finite(cur.get("RSI14", np.nan)) else "N/A",
        "Volume Ratio": f'{cur.get("Volume_Ratio", np.nan):.1f}×' if finite(cur.get("Volume_Ratio", np.nan)) else "N/A",
        "5D Return": f'{cur.get("Return_5d", np.nan):+.1f}%' if finite(cur.get("Return_5d", np.nan)) else "N/A",
        "Sentiment": f'{analysis.get("sentiment",{}).get("compound",0):+.2f}' if analysis.get("sentiment") else "N/A",
        "News Count": analysis.get("sentiment",{}).get("n",0) if analysis.get("sentiment") else 0,
        "P/E Ratio": f'{cur.get("PE_Ratio", np.nan):.1f}' if finite(cur.get("PE_Ratio", np.nan)) else "N/A",
        "RS vs SPY (20d)": f'{float(cur.get("RS_20d_vs_SPY",0) or 0):+.2f}%' if finite(cur.get("RS_20d_vs_SPY", np.nan)) else "N/A",
        "Earnings ≤7d": analysis.get("earnings_in_days", None) if analysis.get("earnings_in_days", None) is not None else ""
    }
    return analysis, row


def scan_enhanced_tickers(tickers: List[str], config: Dict, progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
    results, rows = [], []
    if not tickers: return results, rows
    max_workers = min(6, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, t, config): t for t in tickers}
        for i, fut in enumerate(as_completed(futs)):
            t = futs[fut]
            try:
                res, row = fut.result()
                if res: results.append(res)
                if row: rows.append(row)
            except Exception as e:
                st.warning(f"{t}: {e}")
            if progress_callback:
                progress_callback((i+1)/len(tickers))
            time.sleep(0.02)  # gentle backoff
    results.sort(key=lambda r: r["score"], reverse=True)
    return results, rows

# -------------------- UI --------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.title(APP_TITLE)
    st.caption("Advanced multi-source financial analysis with caching, sentiment, earnings awareness. Not financial advice.")

    if st_autorefresh:
        # 15 minutes (as requested)
        st_autorefresh(interval=15*60*1000, key="auto_refresh_15min")

    settings = load_settings()

    st.sidebar.header("⚙️ Enhanced Configuration")
    market_key = st.sidebar.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
    is_open = cached_is_market_open(market_key)
    mkt = MARKETS[market_key]
    st.sidebar.markdown(f"**Market Status:** {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
    st.sidebar.markdown(f"**Local Time:** {now_tz(mkt['tz']).strftime('%H:%M:%S %Z')}")

    st.sidebar.subheader("📊 Analysis")
    risk_profile = st.sidebar.selectbox("Risk Profile:", ["conservative","balanced","aggressive"],
                                        index=["conservative","balanced","aggressive"].index(settings.get("risk_profile","balanced")))
    lookback_days = st.sidebar.slider("Historical Data (days):", 30, 365, settings.get("lookback_days",120))
    interval = st.sidebar.selectbox("Data Interval:", ["1d","30m"], index=0)

    st.sidebar.subheader("📰 News")
    use_news = st.sidebar.checkbox("Enable News Sentiment", value=True)
    news_days = st.sidebar.slider("News Lookback (days):", 1, 30, settings.get("news_days",7))

    st.sidebar.subheader("📈 Indicators")
    available_ind = list(INDICATOR_CONFIGS.keys())
    selected_ind  = st.sidebar.multiselect("Active Indicators:", available_ind, default=settings.get("indicators",["RSI","MACD","Bollinger"]))

    st.sidebar.subheader("🧪 Extras")
    show_charts = st.sidebar.checkbox("Interactive Charts", value=settings.get("show_charts",True))
    show_correlations = st.sidebar.checkbox("Market Correlations", value=False)
    enable_backtest = st.sidebar.checkbox("Simple Backtest", value=False)
    backtest_hold = st.sidebar.slider("Backtest hold days:", 5, 30, 10)

    # Watchlist
    st.sidebar.subheader("📋 Persistent Watchlist")
    wl = load_watchlist()
    if wl:
        st.sidebar.markdown(f"**Saved ({len(wl)}):** `{', '.join(wl[:8])}{' ...' if len(wl)>8 else ''}`")
    else:
        st.sidebar.info("No stocks in watchlist yet. Add some below!")

    colA, colB = st.sidebar.columns(2)
    with colA:
        new_t = st.text_input("Add Stock:", placeholder="AAPL").strip().upper()
        if st.button("➕ Add") and new_t:
            if 1 <= len(new_t) <= 10 and new_t not in wl:
                wl.append(new_t)
                if save_watchlist(wl): st.rerun()
            else:
                st.sidebar.warning("Invalid or duplicate ticker")
    with colB:
        if wl:
            rem = st.selectbox("Remove:", ["Select..."]+wl)
            if st.button("➖ Remove") and rem!="Select...":
                wl.remove(rem)
                if save_watchlist(wl): st.rerun()
    if st.sidebar.button("📂 Load Popular"):
        popular = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX","AMD","INTC","SPY","QQQ"]
        wl2 = sorted(set((wl or []) + popular))
        if save_watchlist(wl2): st.rerun()
    if wl and st.sidebar.button("🗑️ Clear Watchlist"):
        if save_watchlist([]): st.rerun()

    save_settings({
        "risk_profile": risk_profile,
        "news_sources": ["Google Finance","Yahoo Finance"],
        "indicators": selected_ind,
        "lookback_days": lookback_days,
        "news_days": news_days,
        "show_charts": show_charts,
        "auto_refresh": True,
    })

    if not wl:
        st.warning("🚨 No stocks in watchlist. Add tickers in the sidebar.")
        return

    config = {
        "lookback_days": lookback_days,
        "interval": interval,
        "use_news": use_news,
        "news_days": news_days,
        "risk_profile": risk_profile,
        "indicators": selected_ind,
    }

    if st.button("🚀 Run Enhanced Analysis", type="primary"):
        prog = st.progress(0); info = st.empty()
        def upd(p): prog.progress(p); info.text(f"Analyzing {len(wl)} stocks… {int(p*100)}%")
        with st.spinner("Running comprehensive analysis…"):
            results, table_rows = scan_enhanced_tickers(wl, config, upd)
        prog.empty(); info.empty()

        if not results:
            st.error("❌ No analysis results. Try again.")
            return

        st.header("📊 Analysis Dashboard")
        strong_buy = len([r for r in results if r["signal"]=="BUY" and r["score"]>=80])
        buy_cnt    = len([r for r in results if r["signal"]=="BUY"])
        sell_cnt   = len([r for r in results if r["signal"]=="SELL"])
        avg_conf   = np.mean([r["confidence"] for r in results])
        total      = len(results)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Strong Buy", strong_buy, delta=f"{strong_buy}/{total}")
        c2.metric("Buy Signals", buy_cnt, delta=f"{buy_cnt}/{total}")
        c3.metric("Sell Signals", sell_cnt, delta=f"{sell_cnt}/{total}")
        c4.metric("Avg Confidence", f"{avg_conf:.0f}%")
        c5.metric("Stocks Analyzed", total)

        st.subheader("📈 Detailed Results")
        if table_rows:
            df_res = pd.DataFrame(table_rows)
            st.dataframe(df_res, use_container_width=True)
            st.download_button("📥 Download CSV", df_res.to_csv(index=False).encode("utf-8"),
                               file_name=f"stock_analysis_{dt.date.today():%Y%m%d}.csv", mime="text/csv")

        st.subheader("🎯 Individual Stock Analysis")
        for r in results:
            t = r["ticker"]; sig = r["signal"]; score=r["score"]; conf=r["confidence"]
            badge = "🟢" if sig=="BUY" else ("🔴" if sig=="SELL" else "⚪")
            with st.expander(f"{badge} {t} – {sig} (Score {score}, Confidence {conf}%)"):
                col1,col2 = st.columns([2,1])
                with col1:
                    st.markdown("**Key Signals:**")
                    for i, reason in enumerate(r.get("reasons",[])[:6], 1):
                        st.markdown(f"{i}. {reason}")
                    br = r.get("signals_breakdown",{})
                    if br:
                        st.markdown("**Signal Components:**")
                        for k,v in br.items():
                            if v!=0:
                                st.markdown(f"{'➕' if v>0 else '➖'} {k.title()}: {v:+d}")
                with col2:
                    st.markdown("**Current Data:**")
                    st.markdown(f"Price: **${r['price']:.2f}**")
                    fd = r.get("fundamental_data",{})
                    if fd.get("pe_ratio") is not None: st.markdown(f"P/E: **{fd['pe_ratio']:.1f}**")
                    if fd.get("beta") is not None:     st.markdown(f"Beta: **{fd['beta']:.2f}**")
                    if fd.get("dividend_yield") is not None: st.markdown(f"Dividend: **{fd['dividend_yield']:.2f}%**")
                    if fd.get("sector") and fd.get("sector")!="Unknown": st.markdown(f"Sector: **{fd['sector']}**")
                    if r.get("earnings_in_days") is not None:
                        st.markdown(f"🗓️ Earnings in **{r['earnings_in_days']}** days")
                    sent = r.get("sentiment",{})
                    if sent and sent.get("n",0)>0:
                        emo = "😊" if sent["compound"]>0.1 else ("😐" if sent["compound"]>-0.1 else "😟")
                        st.markdown("**News Sentiment:**")
                        st.markdown(f"{emo} Score: **{sent['compound']:+.2f}** · Articles: **{sent['n']}** · Confidence: **{sent.get('confidence',0)*100:.0f}%**")
                if show_charts:
                    try:
                        df_chart = fetch_price_history(t, lookback_days, interval)
                        if not df_chart.empty:
                            df_chart = compute_enhanced_indicators(df_chart, INDICATOR_CONFIGS, selected_ind)
                            df_chart["RS_20d_vs_SPY"] = relative_strength_20d(df_chart, "SPY")
                            fig = create_enhanced_visualizations(df_chart, t)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart error for {t}: {e}")

                if enable_backtest:
                    try:
                        df_bt = fetch_price_history(t, min(365*3, lookback_days*3), "1d")
                        if not df_bt.empty:
                            df_bt = compute_enhanced_indicators(df_bt, INDICATOR_CONFIGS, selected_ind)
                            res_bt = simple_backtest(df_bt, hold_days=backtest_hold)
                            st.caption(f"🔎 Backtest (hold {backtest_hold}d): buys={res_bt['buy_count']} avg={res_bt['buy_avg']:.2f}% · sells={res_bt['sell_count']} avg={res_bt['sell_avg']:.2f}%")
                    except Exception:
                        pass

        # Correlations across watchlist
        if show_correlations and len(wl) >= 2:
            st.subheader("🌐 Correlation Heatmap (daily returns)")
            try:
                rets = {}
                for t in wl[:20]:  # limit for speed
                    d = fetch_price_history(t, lookback_days, "1d")
                    if d is not None and not d.empty:
                        rets[t] = d["Close"].pct_change().rename(t)
                if rets:
                    R = pd.concat(rets.values(), axis=1).dropna(how="all")
                    if not R.empty:
                        C = R.corr().fillna(0)
                        figC = px.imshow(C, title="Correlation (Pearson)", text_auto=False, aspect="auto")
                        st.plotly_chart(figC, use_container_width=True)
            except Exception as e:
                st.info(f"Correlation unavailable: {e}")

        st.success(f"✅ Analysis complete! Processed {total} stocks.")

if __name__ == "__main__":
    main()
