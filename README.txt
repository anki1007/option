
Quick start

1) Create and activate a venv (optional).
2) pip install -r requirements.txt
3) Set secrets:
   - simplest: export KOTAK_BASE_URL and KOTAK_API_TOKEN (from your NEO dashboard).
   - optional auto-login path: set KOTAK_USERNAME, KOTAK_PASSWORD, KOTAK_CLIENT_CODE, KOTAK_TOTP_SECRET, KOTAK_CONSUMER_KEY and enable checkbox in the app.
4) Run: streamlit run app.py
5) In the sidebar, choose the index, lot size, strike step, and expiries (e.g., 26DEC25).

Tabs:
- Gamma Exposure & Density: shows gamma peaks, neutral strike, total GEX and buy/sell gamma bias.
- Calls vs Puts Volume: sums volumes over ±15 strikes around ATM and plots strike-wise bars.
- Total Greeks: sums Δ Γ ν θ ρ over ±20 strikes (OI-weighted optional). Traffic-light bias: Δ<-10 bearish; Δ>+10 bullish; else sideways.

Notes:
- Kotak NEO response schemas vary; field mapping logic is flexible but you may refine column names for ltp/oi/iv/volume if your account returns different keys.
- Lot sizes and strike steps change; verify and adjust.
- If the quotes endpoint doesn't return IV, the app estimates IV from LTP with bisection; you can replace this with your own IV source for better accuracy.
