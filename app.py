
import os
import math
import time
import json
import typing as T
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import pyotp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================
# ⚠️ SECURITY & SETUP
# ----------------------------
# Put your secrets in environment variables or in Streamlit secrets.
# Required:
#   KOTAK_BASE_URL
#   KOTAK_API_TOKEN           (preferred simple path)
# Optional (if you want auto-login with TOTP; adjust endpoints as needed):
#   KOTAK_USERNAME
#   KOTAK_PASSWORD
#   KOTAK_CLIENT_CODE
#   KOTAK_TOTP_SECRET
#   KOTAK_CONSUMER_KEY
#
# Never hardcode secrets in this file.
# ============================

st.set_page_config(page_title="Gamma Exposure & Greeks – Kotak NEO", layout="wide")

# ---------- Utilities ----------

def get_secret(name: str, default: str | None = None) -> str | None:
    """Fetch from st.secrets or env; do not crash if missing."""
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

def to_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return str(x).strip().lower() in {"1","true","yes","y","on"}

def dtonow():
    return datetime.now(timezone.utc)

# ---------- Kotak NEO API Client (quotes endpoint + optional login) ----------

class KotakClient:
    def __init__(self, base_url: str, api_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._api_token = api_token  # static token path
        self._bearer = None          # dynamic token path
        self._last_login = None

    @property
    def headers(self) -> dict:
        token = self._bearer or self._api_token
        if not token:
            raise RuntimeError("Missing API token/bearer. Provide KOTAK_API_TOKEN or enable auto-login.")
        return {
            "Authorization": token,
            "Content-Type": "application/json"
        }

    # NOTE: This is a placeholder flow; adjust endpoints to Kotak NEO auth if you want auto-login.
    def autologin(self) -> None:
        user = get_secret("KOTAK_USERNAME")
        pwd = get_secret("KOTAK_PASSWORD")
        client_code = get_secret("KOTAK_CLIENT_CODE")
        totp_secret = get_secret("KOTAK_TOTP_SECRET")
        consumer_key = get_secret("KOTAK_CONSUMER_KEY")
        if not all([user, pwd, client_code, totp_secret, consumer_key]):
            raise RuntimeError("Auto-login requires KOTAK_USERNAME, KOTAK_PASSWORD, KOTAK_CLIENT_CODE, KOTAK_TOTP_SECRET, KOTAK_CONSUMER_KEY.")

        otp = pyotp.TOTP(totp_secret).now()
        # Replace these paths with the actual Kotak NEO login endpoints if needed.
        # Example sketch (NOT official):
        try:
            login_url = f"{self.base_url}/session/1.0/login"
            payload = {
                "username": user,
                "password": pwd,
                "client_code": client_code,
                "consumer_key": consumer_key,
                "totp": otp,
            }
            resp = requests.post(login_url, headers={"Content-Type":"application/json"}, data=json.dumps(payload), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Expecting something like data["access_token"]
            self._bearer = data.get("access_token") or data.get("token") or None
            if not self._bearer:
                raise RuntimeError("Auto-login succeeded but no bearer token field found. Inspect response.")
            self._last_login = dtonow()
        except Exception as e:
            raise RuntimeError(f"Auto-login failed: {e}")

    def ensure_auth(self):
        if self._api_token:
            return
        if not self._bearer:
            self.autologin()

    def get_quotes_by_neosymbol(self, queries: list[str], filter_name: str = "all") -> dict:
        """Use the provided endpoint format:
           GET <Base URL>/script-details/1.0/quotes/neosymbol/<query>[,<query>][/<filter_name>]
           where each query like: nse_fo|NIFTY 50 26DEC25 25000 CE
           For indices spot, examples seen: nse_cm|Nifty 50
        """
        self.ensure_auth()
        path = "/script-details/1.0/quotes/neosymbol/"
        q = ",".join(queries)
        url = f"{self.base_url}{path}{q}"
        if filter_name:
            url += f"/{filter_name}"
        r = requests.get(url, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

# ---------- Greeks (Black–Scholes helpers) ----------

SQRT_2PI = math.sqrt(2*math.pi)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x)/SQRT_2PI

def _norm_cdf(x: float) -> float:
    # Abramowitz & Stegun approximation
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = k*(0.319381530 + k*(-0.356563782 + k*(1.781477937 + k*(-1.821255978 + 1.330274429*k))))
    cnd = 1.0 - _norm_pdf(x) * poly
    return cnd if x >= 0 else 1.0 - cnd

def bs_d1(S, K, r, q, sigma, T):
    return (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))

def bs_d2(d1, sigma, T):
    return d1 - sigma*math.sqrt(T)

def bs_price(is_call, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S*math.exp(-q*T) - K*math.exp(-r*T)) if is_call else (K*math.exp(-r*T) - S*math.exp(-q*T)))
    d1 = bs_d1(S,K,r,q,sigma,T)
    d2 = bs_d2(d1,sigma,T)
    if is_call:
        return S*math.exp(-q*T)*_norm_cdf(d1) - K*math.exp(-r*T)*_norm_cdf(d2)
    else:
        return K*math.exp(-r*T)*_norm_cdf(-d2) - S*math.exp(-q*T)*_norm_cdf(-d1)

def bs_delta(is_call, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return math.exp(-q*T)*(_norm_cdf(d1) if is_call else _norm_cdf(d1)-1.0)

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return math.exp(-q*T) * _norm_pdf(d1) / (S*sigma*math.sqrt(T))

def bs_theta(is_call, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    d2 = bs_d2(d1,sigma,T)
    first = - (S*math.exp(-q*T) * _norm_pdf(d1) * sigma) / (2*math.sqrt(T))
    if is_call:
        return first + q*S*math.exp(-q*T)*_norm_cdf(d1) - r*K*math.exp(-r*T)*_norm_cdf(d2)
    else:
        return first - q*S*math.exp(-q*T)*_norm_cdf(-d1) + r*K*math.exp(-r*T)*_norm_cdf(-d2)

def bs_vega(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return S*math.exp(-q*T)*_norm_pdf(d1)*math.sqrt(T)

def bs_rho(is_call, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    d2 = bs_d2(d1,sigma,T)
    if is_call:
        return K*T*math.exp(-r*T)*_norm_cdf(d2)
    else:
        return -K*T*math.exp(-r*T)*_norm_cdf(-d2)

def implied_vol(is_call, S, K, r, q, T, price, tol=1e-5, max_iter=100):
    # Simple Brent-like bisection bounds
    if price <= 0:
        return np.nan
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        pm = bs_price(is_call,S,K,r,q,mid,T)
        if abs(pm - price) < tol:
            return mid
        if pm > price:
            high = mid
        else:
            low = mid
    return np.nan

# ---------- Domain helpers ----------

INDEX_INFO = {
    "NIFTY": {
        "neosymbol": "nse_cm|Nifty 50",
        "lot_size": 50,     # verify current lot size
        "multiplier": 1,
        "exchange": "nse_fo",
        "label": "NIFTY 50"
    },
    "BANKNIFTY": {
        "neosymbol": "nse_cm|Nifty Bank",
        "lot_size": 15,     # verify current lot size
        "multiplier": 1,
        "exchange": "nse_fo",
        "label": "BANKNIFTY"
    },
    "SENSEX": {
        "neosymbol": "bse_cm|Sensex",
        "lot_size": 10,     # verify current lot size on BSE
        "multiplier": 1,
        "exchange": "bse_fo",
        "label": "SENSEX"
    }
}

def nearest_atm_strike(spot: float, step: int = 50) -> int:
    return int(round(spot / step) * step)

def build_option_neosymbol(exchange: str, underlying: str, expiry: str, strike: int, opt_type: str) -> str:
    # Example format (adjust to exact Kotak format for F&O):
    # nse_fo|NIFTY 26DEC25 25000 CE
    return f"{exchange}|{underlying} {expiry} {strike} {opt_type}"

def derive_time_to_expiry(expiry_dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    T = (expiry_dt - now).total_seconds() / (365.0*24*3600.0)
    return max(T, 1e-6)

def safe_number(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (int,float,np.floating)):
            return float(x)
        return float(str(x).replace(",",""))
    except:
        return default

# ---------- Data acquisition layer ----------

@st.cache_data(ttl=10.0, show_spinner=False)
def fetch_underlying_quote(client: KotakClient, neosymbol: str) -> dict:
    data = client.get_quotes_by_neosymbol([neosymbol], "all")
    return data

@st.cache_data(ttl=10.0, show_spinner=False)
def fetch_option_chain(client: KotakClient, exchange: str, underlying_label: str, expiry_label: str, center_strike: int, width: int, step: int) -> pd.DataFrame:
    queries = []
    strikes = list(range(center_strike - width*step, center_strike + (width+1)*step, step))
    for k in strikes:
        for t in ("CE","PE"):
            q = build_option_neosymbol(exchange, underlying_label, expiry_label, k, t)
            queries.append(q)
    raw = client.get_quotes_by_neosymbol(queries, "all")
    # Normalize; the shape of 'raw' depends on Kotak response. We'll attempt generic normalization.
    rows = []
    def flatten(obj, prefix=""):
        out = {}
        if isinstance(obj, dict):
            for k,v in obj.items():
                sub = flatten(v, f"{prefix}{k}.")
                if sub:
                    out.update(sub)
                else:
                    out[f"{prefix}{k}"] = v
        return out
    # Expect either {"data":[{...}, {...}]} or list
    container = raw.get("data", raw)
    if isinstance(container, dict):
        container = container.get("quotes") or container.get("items") or []
    if not isinstance(container, (list,tuple)):
        container = []
    for item in container:
        flat = flatten(item)
        flat.update(item if isinstance(item, dict) else {})
        rows.append(flat)
    df = pd.DataFrame(rows)
    # Try to parse essential fields with flexible column names:
    # ltp, lastPrice, closePrice; volume; openInterest; strikePrice; optionType; expiryDate; impliedVolatility
    colmap = {
        "ltp": ["ltp","lastPrice","LastPrice","price.ltp","trade.ltp"],
        "volume": ["volume","totalTradedVolume","trade.volume"],
        "oi": ["openInterest","oi","open_interest"],
        "iv": ["impliedVolatility","iv","option.greeks.iv"],
        "strike": ["strikePrice","strike","option.strike"],
        "opt_type": ["optionType","optType","option.type"],
        "expiry": ["expiryDate","expiry","option.expiry"],
        "symbol": ["symbol","tradingsymbol","symbolName","instrumentName"]
    }
    def pick(row, keys, default=np.nan):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return default
    if df.empty:
        return df
    df["ltp"] = df.apply(lambda r: safe_number(pick(r, colmap["ltp"])), axis=1)
    df["volume"] = df.apply(lambda r: safe_number(pick(r, colmap["volume"], 0.0), 0.0), axis=1)
    df["oi"] = df.apply(lambda r: safe_number(pick(r, colmap["oi"], 0.0), 0.0), axis=1)
    df["iv"] = df.apply(lambda r: safe_number(pick(r, colmap["iv"])), axis=1)
    df["strike"] = df.apply(lambda r: int(safe_number(pick(r, colmap["strike"], np.nan))), axis=1)
    df["opt_type"] = df.apply(lambda r: str(pick(r, colmap["opt_type"], "CE")).upper(), axis=1)
    df["expiry"] = df.apply(lambda r: str(pick(r, colmap["expiry"], expiry_label)), axis=1)
    df["symbol"] = df.apply(lambda r: pick(r, colmap["symbol"], ""), axis=1)
    df = df.dropna(subset=["strike","ltp"]).reset_index(drop=True)
    return df

# ---------- Analytics ----------

def compute_greeks_table(df: pd.DataFrame, spot: float, r: float, q: float, expiry_dt: datetime) -> pd.DataFrame:
    T = derive_time_to_expiry(expiry_dt)
    rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        is_call = row["opt_type"] == "CE"
        price = float(row["ltp"])
        iv = row.get("iv", np.nan)
        if not (isinstance(iv, (int,float,np.floating)) and iv>0 and iv<5):
            iv = implied_vol(is_call, spot, K, r, q, T, price) or np.nan
        if not (isinstance(iv, (int,float,np.floating)) and iv>0 and iv<5):
            # fallback small sigma to avoid NaNs
            iv = 0.20
        delta = bs_delta(is_call, spot, K, r, q, iv, T)
        gamma = bs_gamma(spot, K, r, q, iv, T)
        vega  = bs_vega(spot, K, r, q, iv, T)
        theta = bs_theta(is_call, spot, K, r, q, iv, T)
        rho   = bs_rho(is_call, spot, K, r, q, iv, T)
        rows.append({
            "strike": int(K),
            "type": "CALL" if is_call else "PUT",
            "ltp": price,
            "iv": iv,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
            "volume": float(row.get("volume", 0.0)),
            "oi": float(row.get("oi", 0.0)),
            "expiry": row.get("expiry","")
        })
    gdf = pd.DataFrame(rows).sort_values(["strike","type"])
    return gdf

def compute_gex_gamma_density(gdf: pd.DataFrame, lot_size: int, multiplier: int, spot: float) -> pd.DataFrame:
    # GEX ≈ Σ gamma * OI * contract_multiplier * S^2 * lot_size
    # Sign convention: calls and puts both add positive gamma; exposure sign derives from netting hedging flows.
    # We will compute per strike 'gamma_exposure' and allow visualization of peaks.
    g = gdf.copy()
    g["gamma_contract"] = g["gamma"] * (spot**2)
    g["gex_oi"] = g["gamma_contract"] * g["oi"] * lot_size * multiplier
    # Aggregate by strike; density is simply gex_oi per strike.
    density = g.groupby("strike", as_index=False)["gex_oi"].sum().rename(columns={"gex_oi":"gamma_exposure"})
    # Cumulative for zero-cross/neutral (approx)
    density = density.sort_values("strike").reset_index(drop=True)
    density["cum_gex"] = density["gamma_exposure"].cumsum()
    return density

def classify_bias(total_gex: float, pos_threshold: float, neg_threshold: float) -> str:
    # Simplified bias: positive GEX -> "buy gamma"; negative -> "sell gamma"
    if total_gex > pos_threshold: return "Long Gamma (buy gamma)"
    if total_gex < -abs(neg_threshold): return "Short Gamma (sell gamma)"
    return "Gamma Neutral/Range"

def aggregate_greeks(gdf: pd.DataFrame, strikes_window: int, center_strike: int, use_oi_weighted: bool, lot_size: int) -> dict:
    window = gdf[(gdf["strike"] >= center_strike - strikes_window) & (gdf["strike"] <= center_strike + strikes_window)]
    if use_oi_weighted:
        w = window.copy()
        w["w"] = w["oi"] * lot_size
        tot = {
            "delta": (w["delta"]*w["w"]).sum(),
            "gamma": (w["gamma"]*w["w"]).sum(),
            "vega":  (w["vega"] *w["w"]).sum(),
            "theta": (w["theta"]*w["w"]).sum(),
            "rho":   (w["rho"]  *w["w"]).sum(),
        }
    else:
        tot = {
            "delta": window["delta"].sum(),
            "gamma": window["gamma"].sum(),
            "vega":  window["vega"].sum(),
            "theta": window["theta"].sum(),
            "rho":   window["rho"].sum(),
        }
    return tot

# ---------- UI ----------

st.title("Institutional Gamma Exposure & Greeks Dashboard (Kotak NEO)")

# Sidebar: config
with st.sidebar:
    st.subheader("Connection")
    base_url = st.text_input("KOTAK_BASE_URL", value=get_secret("KOTAK_BASE_URL","https://api.kotaksecurities.com"))
    api_token = st.text_input("KOTAK_API_TOKEN (preferred)", value=get_secret("KOTAK_API_TOKEN",""), type="password")
    enable_autologin = st.checkbox("Enable Auto-login with TOTP (optional)", value=False)
    st.caption("If enabled, provide username/password/client code/TOTP secret/consumer key via Streamlit secrets or environment variables.")
    st.divider()

    st.subheader("Market")
    index_key = st.selectbox("Index", options=list(INDEX_INFO.keys()), index=0, format_func=lambda k: INDEX_INFO[k]["label"])
    lot_size = st.number_input("Lot Size", value=int(INDEX_INFO[index_key]["lot_size"]), step=1)
    step = st.number_input("Strike Step", value=50, step=50)
    rfr = st.number_input("Risk-free Rate (annual, e.g. 0.0675)", value=0.0675, format="%.4f")
    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, format="%.4f")
    pos_thr = st.number_input("Positive GEX threshold (for bias)", value=1e9, format="%.1f")
    neg_thr = st.number_input("Negative GEX threshold (for bias)", value=1e9, format="%.1f")

    st.subheader("Strikes Windows")
    width_gamma = st.number_input("± strikes for Gamma Density", value=20, step=5)
    width_volcomp = st.number_input("± strikes for Volume Compare (Calls vs Puts)", value=15, step=1)
    width_greeks = st.number_input("± strikes for Greeks Sum", value=20, step=1)
    use_oi_weighted = st.checkbox("Use OI-weighted totals (recommended)", value=True)

client = KotakClient(base_url, api_token.strip() or None)
if not api_token and enable_autologin:
    try:
        client.autologin()
        st.success("Auto-login OK")
    except Exception as e:
        st.error(str(e))

# Spot quote
index_info = INDEX_INFO[index_key]
spot_json = {}
spot_value = np.nan
try:
    spot_json = fetch_underlying_quote(client, index_info["neosymbol"])
    # Attempt to parse spot
    # Similar flexible parsing as before
    d = spot_json.get("data", spot_json)
    if isinstance(d, dict):
        d = d.get("quotes") or d.get("items") or [d]
    if isinstance(d, list) and d:
        item = d[0]
    else:
        item = spot_json
    possible_spot_keys = ["lastPrice","ltp","price","price.ltp","trade.ltp","index.lastPrice"]
    for k in possible_spot_keys:
        val = item.get(k) if isinstance(item, dict) else None
        if val is not None:
            spot_value = safe_number(val)
            break
except Exception as e:
    st.error(f"Failed to fetch underlying: {e}")

if np.isnan(spot_value):
    st.stop()

st.markdown(f"**{index_info['label']} Spot:** `{spot_value:,.2f}`")

# Expiry selection input (free text, e.g., '26DEC25' or according to your API)
expiry_label = st.text_input("Expiry label (e.g., 26DEC25)", value="26DEC25")
# Parse expiry date roughly: DDMMMYY -> convert to datetime at 15:30 IST; here use UTC 10:00 approximate
try:
    expiry_dt = datetime.strptime(expiry_label, "%d%b%y").replace(tzinfo=timezone.utc)
except:
    # fallback to a near future date
    expiry_dt = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)

atm = nearest_atm_strike(spot_value, step=int(step))

# Chain fetch
try:
    chain_df = fetch_option_chain(client, index_info["exchange"], index_info["label"].split()[0], expiry_label, atm, width=int(max(width_gamma,width_greeks,width_volcomp)), step=int(step))
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")
    st.stop()

if chain_df.empty:
    st.warning("Empty option chain. Check expiry label/strike format and API permissions.")
    st.stop()

# Compute Greeks
gdf = compute_greeks_table(chain_df, spot_value, r=rfr, q=dividend_yield, expiry_dt=expiry_dt)

tab1, tab2, tab3 = st.tabs(["Gamma Exposure & Density", "Calls vs Puts Volume (±15 strikes)", "Total Greeks (±20 strikes)"])

with tab1:
    st.subheader("Gamma Density & Peaks")
    dens = compute_gex_gamma_density(gdf, lot_size=lot_size, multiplier=index_info["multiplier"], spot=spot_value)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dens["strike"], y=dens["gamma_exposure"], name="Gamma Exposure"))
    # Mark peaks
    if len(dens) > 2:
        max_idx = dens["gamma_exposure"].idxmax()
        min_idx = dens["gamma_exposure"].idxmin()
        fig.add_trace(go.Scatter(x=[dens.loc[max_idx,"strike"]], y=[dens.loc[max_idx,"gamma_exposure"]], mode="markers+text", text=["Max +Gamma"], textposition="top center", name="Peak +"))
        fig.add_trace(go.Scatter(x=[dens.loc[min_idx,"strike"]], y=[dens.loc[min_idx,"gamma_exposure"]], mode="markers+text", text=["Max -Gamma"], textposition="bottom center", name="Peak -"))
    fig.update_layout(xaxis_title="Strike", yaxis_title="Gamma Exposure (approx)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Neutral Gamma (zero-cross)")
    # Find strike where cum_gex crosses zero
    zero_strike = None
    sgn = np.sign(dens["cum_gex"].values)
    for i in range(1, len(sgn)):
        if sgn[i] == 0:
            zero_strike = dens.loc[i, "strike"]
            break
        if sgn[i] != sgn[i-1]:
            # interpolate
            x1, y1 = dens.loc[i-1,"strike"], dens.loc[i-1,"cum_gex"]
            x2, y2 = dens.loc[i,"strike"], dens.loc[i,"cum_gex"]
            if (y2 - y1) != 0:
                zero_strike = int(round(x1 - y1*(x2-x1)/(y2-y1)))
            else:
                zero_strike = dens.loc[i,"strike"]
            break
    st.write(f"**Estimated Gamma Neutral Strike:** {zero_strike if zero_strike else 'Not found in window'}")

    total_gex = dens["gamma_exposure"].sum()
    bias = classify_bias(total_gex, pos_threshold=pos_thr, neg_threshold=neg_thr)
    st.metric("Total GEX (sum)", f"{total_gex:,.0f}", help="Sum of gamma exposure across strikes in window")
    st.metric("Bias", bias)

    st.markdown("""
**Legend**  
- **Long Gamma (buy gamma)**: positive exposure; market makers hedge by buying dips/selling rips → dampened moves.  
- **Short Gamma (sell gamma)**: negative exposure; hedging amplifies moves → higher volatility.  
- **Neutral**: around zero; directional bias from gamma is limited.
""")

with tab2:
    st.subheader("Total Call vs Put Volumes (±15 strikes from ATM)")
    window = gdf[(gdf["strike"] >= atm - width_volcomp*step) & (gdf["strike"] <= atm + width_volcomp*step)]
    vol_calls = window.loc[window["type"]=="CALL","volume"].sum()
    vol_puts  = window.loc[window["type"]=="PUT","volume"].sum()

    colA, colB = st.columns(2)
    with colA:
        st.metric("Total Call Volume", f"{vol_calls:,.0f}")
    with colB:
        st.metric("Total Put Volume", f"{vol_puts:,.0f}")

    fig2 = go.Figure()
    sgrp = window.groupby(["strike","type"], as_index=False)["volume"].sum()
    sgrp_pivot = sgrp.pivot(index="strike", columns="type", values="volume").fillna(0.0)
    sgrp_pivot = sgrp_pivot.reset_index()
    fig2.add_trace(go.Bar(x=sgrp_pivot["strike"], y=sgrp_pivot.get("CALL", pd.Series(0)), name="CALLs"))
    fig2.add_trace(go.Bar(x=sgrp_pivot["strike"], y=sgrp_pivot.get("PUT", pd.Series(0)), name="PUTs"))
    fig2.update_layout(barmode="group", xaxis_title="Strike", yaxis_title="Volume")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Aggregated Greeks (±20 strikes from ATM)")
    totals = aggregate_greeks(gdf, strikes_window=width_greeks*step, center_strike=atm, use_oi_weighted=use_oi_weighted, lot_size=lot_size)
    df_tot = pd.DataFrame([totals], index=["Totals"]).T.rename(columns={"Totals":"Value"})
    st.dataframe(df_tot)

    # Interpretation: user requested -10 bearish, +10 bullish, between sideways
    # We'll apply to TOTAL DELTA only for the traffic-light verdict.
    verdict = "Sideways"
    if totals["delta"] > 10: verdict = "Bullish"
    elif totals["delta"] < -10: verdict = "Bearish"

    st.metric("Bias by Total Δ", verdict, help="Rule: Δ < -10 → Bearish, -10≤Δ≤+10 → Sideways, Δ > +10 → Bullish")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=list(totals.keys()), y=list(totals.values())))
    fig3.update_layout(xaxis_title="Greek", yaxis_title="Aggregated Value")
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Note: Ensure your expiry string and neosymbol format matches Kotak NEO. Lot sizes and strike steps can change; verify periodically.")
