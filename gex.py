import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import pyotp
from neo_api_client import NeoAPI
import json

# Page Configuration
st.set_page_config(
    page_title="Institutional Gamma & Options Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional look
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}

# Kotak Neo API Credentials
CREDENTIALS = {
    "consumer_key": "9cc06478-fc24-4344-bf37-112522125cd9",
    "consumer_secret": "",  # Add if you have it
    "mobile": "8800616007",
    "password": "Jan@2026",
    "totp_key": "VJ5TLV5QMUH5OCW3CFQG4ZL5OU",
    "environment": "prod"
}

def generate_totp():
    """Generate TOTP for 2FA"""
    totp = pyotp.TOTP(CREDENTIALS['totp_key'])
    return totp.now()

def auto_login():
    """Auto-login to Kotak Neo API"""
    try:
        with st.spinner("Logging in to Kotak Neo API..."):
            client = NeoAPI(
                consumer_key=CREDENTIALS['consumer_key'],
                consumer_secret=CREDENTIALS.get('consumer_secret', ''),
                environment=CREDENTIALS['environment']
            )
            
            # Step 1: Login
            client.login(
                mobilenumber=CREDENTIALS['mobile'],
                password=CREDENTIALS['password']
            )
            
            # Step 2: Generate and submit OTP
            otp = generate_totp()
            client.session_2fa(OTP=otp)
            
            st.session_state.client = client
            st.session_state.logged_in = True
            st.success("✅ Successfully logged in!")
            return client
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def calculate_gamma_exposure(spot_price, strikes, call_oi, put_oi, call_gamma, put_gamma):
    """Calculate Gamma Exposure (GEX) for options"""
    # Gamma Exposure = Gamma * Open Interest * Contract Multiplier * Spot^2 / 100
    contract_multiplier = 50  # For Nifty/Bank Nifty
    
    call_gex = call_gamma * call_oi * contract_multiplier * spot_price / 100
    put_gex = put_gamma * put_oi * contract_multiplier * spot_price / 100
    
    net_gex = call_gex - put_gex  # Calls are positive, Puts are negative
    
    return net_gex, call_gex, put_gex

def get_options_chain(client, symbol, index_symbol):
    """Fetch options chain data from Kotak Neo API"""
    try:
        # Get spot price
        spot_data = client.quotes(
            instrument_tokens=[{"instrument_token": index_symbol, "exchange_segment": "nse_cm"}],
            isIndex=True
        )
        spot_price = float(spot_data['message'][0]['last_traded_price'])
        
        # Get ATM strike
        strike_diff = 50 if 'Nifty 50' in index_symbol or 'Bank' in index_symbol else 100
        atm_strike = round(spot_price / strike_diff) * strike_diff
        
        # Generate strike range (20 strikes on each side)
        strikes = [atm_strike + (i * strike_diff) for i in range(-20, 21)]
        
        # Fetch options data for all strikes
        options_data = {
            'strike': [],
            'call_oi': [],
            'put_oi': [],
            'call_volume': [],
            'put_volume': [],
            'call_ltp': [],
            'put_ltp': [],
            'call_delta': [],
            'put_delta': [],
            'call_gamma': [],
            'put_gamma': [],
            'call_vega': [],
            'put_vega': [],
            'call_theta': [],
            'put_theta': [],
            'call_rho': [],
            'put_rho': []
        }
        
        # Simulated data for demonstration (replace with actual API calls)
        for strike in strikes:
            options_data['strike'].append(strike)
            
            # Generate realistic synthetic data
            moneyness = abs(strike - spot_price) / spot_price
            
            # Open Interest
            base_oi = np.random.randint(50000, 500000)
            options_data['call_oi'].append(int(base_oi * (1 - moneyness)))
            options_data['put_oi'].append(int(base_oi * (1 + moneyness)))
            
            # Volume
            options_data['call_volume'].append(np.random.randint(10000, 100000))
            options_data['put_volume'].append(np.random.randint(10000, 100000))
            
            # LTP
            options_data['call_ltp'].append(max(1, 300 * np.exp(-moneyness * 5)))
            options_data['put_ltp'].append(max(1, 300 * np.exp(-moneyness * 5)))
            
            # Greeks (simplified calculations)
            gamma_val = 0.005 * np.exp(-moneyness * 10)
            options_data['call_gamma'].append(gamma_val)
            options_data['put_gamma'].append(gamma_val)
            
            if strike < spot_price:
                options_data['call_delta'].append(0.5 + 0.3 * (1 - moneyness))
                options_data['put_delta'].append(-0.5 + 0.3 * moneyness)
            else:
                options_data['call_delta'].append(0.5 - 0.3 * moneyness)
                options_data['put_delta'].append(-0.5 - 0.3 * (1 - moneyness))
            
            options_data['call_vega'].append(np.random.uniform(5, 15))
            options_data['put_vega'].append(np.random.uniform(5, 15))
            
            options_data['call_theta'].append(np.random.uniform(-5, -1))
            options_data['put_theta'].append(np.random.uniform(-5, -1))
            
            options_data['call_rho'].append(np.random.uniform(1, 5))
            options_data['put_rho'].append(np.random.uniform(-5, -1))
        
        return pd.DataFrame(options_data), spot_price
        
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return None, None

def plot_gamma_exposure(df, spot_price, symbol):
    """Create Gamma Exposure visualization"""
    # Calculate GEX
    net_gex, call_gex, put_gex = calculate_gamma_exposure(
        spot_price,
        df['strike'].values,
        df['call_oi'].values,
        df['put_oi'].values,
        df['call_gamma'].values,
        df['put_gamma'].values
    )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Net Gamma Exposure', 'Gamma Density'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Net GEX bar chart
    colors = ['red' if x < 0 else 'green' for x in net_gex]
    fig.add_trace(
        go.Bar(x=df['strike'], y=net_gex, name='Net GEX', marker_color=colors),
        row=1, col=1
    )
    
    # Add spot price line
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", 
                  annotation_text=f"Spot: {spot_price:.2f}", row=1, col=1)
    
    # Gamma density
    fig.add_trace(
        go.Scatter(x=df['strike'], y=call_gex, name='Call Gamma', 
                   fill='tozeroy', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['strike'], y=abs(put_gex), name='Put Gamma', 
                   fill='tozeroy', line=dict(color='red')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Gamma Exposure (MM)", row=1, col=1)
    fig.update_yaxes(title_text="Gamma Density", row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig, net_gex, call_gex, put_gex

def determine_bias(net_gex):
    """Determine market bias based on gamma exposure"""
    total_gex = np.sum(net_gex)
    
    if total_gex > 10:
        return "🟢 BULLISH", "green"
    elif total_gex < -10:
        return "🔴 BEARISH", "red"
    else:
        return "🟡 NEUTRAL", "orange"

def main():
    # Header
    st.markdown('<div class="main-header">📊 Institutional Gamma & Options Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔐 Authentication")
        
        if not st.session_state.logged_in:
            if st.button("🔑 Auto Login", type="primary"):
                auto_login()
        else:
            st.success("✅ Connected")
            if st.button("🔓 Logout"):
                st.session_state.logged_in = False
                st.session_state.client = None
                st.rerun()
        
        st.divider()
        
        st.title("⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 60, 5)
        
        st.divider()
        
        st.title("📈 Select Index")
        index_selection = st.selectbox(
            "Choose Index",
            ["Nifty 50", "Nifty Bank", "SENSEX"]
        )
    
    # Main content
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login using the sidebar to access the dashboard")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Gamma Exposure & Density",
        "📈 Call/Put Volume Analysis",
        "🎯 Greeks Analysis"
    ])
    
    # Tab 1: Gamma Exposure
    with tab1:
        st.subheader(f"🎯 {index_selection} - Gamma Exposure Dashboard")
        
        # Fetch data
        client = st.session_state.client
        df, spot_price = get_options_chain(client, index_selection, index_selection)
        
        if df is not None and spot_price is not None:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Spot Price", f"₹{spot_price:.2f}")
            
            # Calculate metrics
            net_gex, call_gex, put_gex = calculate_gamma_exposure(
                spot_price, df['strike'].values, df['call_oi'].values,
                df['put_oi'].values, df['call_gamma'].values, df['put_gamma'].values
            )
            
            total_net_gex = np.sum(net_gex)
            bias, bias_color = determine_bias(net_gex)
            
            with col2:
                st.metric("Net Gamma Exposure", f"{total_net_gex:.2f}M")
            
            with col3:
                st.metric("Call Gamma", f"{np.sum(call_gex):.2f}M", delta="Positive")
            
            with col4:
                st.metric("Put Gamma", f"{np.sum(put_gex):.2f}M", delta="Negative", delta_color="inverse")
            
            # Bias indicator
            st.markdown(f"### Market Bias: <span style='color:{bias_color}; font-size:24px; font-weight:bold'>{bias}</span>", 
                       unsafe_allow_html=True)
            
            # Plot
            fig, _, _, _ = plot_gamma_exposure(df, spot_price, index_selection)
            st.plotly_chart(fig, use_container_width=True)
            
            # Gamma peaks
            st.subheader("🔝 Top Gamma Peaks")
            peak_indices = np.argsort(np.abs(net_gex))[-5:][::-1]
            peak_df = pd.DataFrame({
                'Strike': df.iloc[peak_indices]['strike'].values,
                'Net GEX (MM)': net_gex[peak_indices],
                'Type': ['Resistance' if x > 0 else 'Support' for x in net_gex[peak_indices]]
            })
            st.dataframe(peak_df, use_container_width=True)
    
    # Tab 2: Call/Put Volume Analysis
    with tab2:
        st.subheader(f"📊 {index_selection} - Call vs Put Volume Analysis")
        
        if df is not None:
            # Filter for 15 strikes ITM to OTM
            atm_idx = (df['strike'] - spot_price).abs().argmin()
            analysis_df = df.iloc[max(0, atm_idx-15):min(len(df), atm_idx+16)].copy()
            
            # Calculate totals
            total_call_vol = analysis_df['call_volume'].sum()
            total_put_vol = analysis_df['put_volume'].sum()
            pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Call Volume", f"{total_call_vol:,.0f}")
            with col2:
                st.metric("Total Put Volume", f"{total_put_vol:,.0f}")
            with col3:
                pcr_color = "red" if pcr_volume > 1.2 else "green" if pcr_volume < 0.8 else "orange"
                st.markdown(f"**PCR (Volume):** <span style='color:{pcr_color}; font-size:24px'>{pcr_volume:.2f}</span>", 
                           unsafe_allow_html=True)
            
            # Volume chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=analysis_df['strike'],
                y=analysis_df['call_volume'],
                name='Call Volume',
                marker_color='green'
            ))
            fig.add_trace(go.Bar(
                x=analysis_df['strike'],
                y=analysis_df['put_volume'],
                name='Put Volume',
                marker_color='red'
            ))
            
            fig.add_vline(x=spot_price, line_dash="dash", line_color="blue",
                         annotation_text=f"Spot: {spot_price:.2f}")
            
            fig.update_layout(
                title="Call vs Put Volume Distribution",
                xaxis_title="Strike Price",
                yaxis_title="Volume",
                barmode='group',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("📝 Interpretation")
            if pcr_volume > 1.2:
                st.info("🔴 **Bearish Signal**: Put volume significantly exceeds call volume (PCR > 1.2)")
            elif pcr_volume < 0.8:
                st.info("🟢 **Bullish Signal**: Call volume significantly exceeds put volume (PCR < 0.8)")
            else:
                st.info("🟡 **Neutral Signal**: Call and put volumes are relatively balanced (0.8 < PCR < 1.2)")
    
    # Tab 3: Greeks Analysis
    with tab3:
        st.subheader(f"🎯 {index_selection} - Greeks Analysis (20 Strikes ITM to OTM)")
        
        if df is not None:
            # Filter for 20 strikes each side
            atm_idx = (df['strike'] - spot_price).abs().argmin()
            greeks_df = df.iloc[max(0, atm_idx-20):min(len(df), atm_idx+21)].copy()
            
            # Calculate total Greeks
            total_delta = greeks_df['call_delta'].sum() + greeks_df['put_delta'].sum()
            total_gamma = greeks_df['call_gamma'].sum() + greeks_df['put_gamma'].sum()
            total_vega = greeks_df['call_vega'].sum() + greeks_df['put_vega'].sum()
            total_theta = greeks_df['call_theta'].sum() + greeks_df['put_theta'].sum()
            total_rho = greeks_df['call_rho'].sum() + greeks_df['put_rho'].sum()
            
            # Display Greeks metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                delta_color = "green" if total_delta > 10 else "red" if total_delta < -10 else "orange"
                st.markdown(f"**Total Δ Delta**<br><span style='color:{delta_color}; font-size:28px'>{total_delta:.2f}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Total Γ Gamma**<br><span style='font-size:28px'>{total_gamma:.4f}</span>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Total ν Vega**<br><span style='font-size:28px'>{total_vega:.2f}</span>", 
                           unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"**Total Θ Theta**<br><span style='font-size:28px'>{total_theta:.2f}</span>", 
                           unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"**Total ρ Rho**<br><span style='font-size:28px'>{total_rho:.2f}</span>", 
                           unsafe_allow_html=True)
            
            st.divider()
            
            # Greeks visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Delta Distribution', 'Gamma Distribution', 
                               'Vega Distribution', 'Theta Distribution'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Delta
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['call_delta'], 
                                    name='Call Δ', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['put_delta'], 
                                    name='Put Δ', line=dict(color='red')), row=1, col=1)
            
            # Gamma
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['call_gamma'], 
                                    name='Call Γ', line=dict(color='green')), row=1, col=2)
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['put_gamma'], 
                                    name='Put Γ', line=dict(color='red')), row=1, col=2)
            
            # Vega
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['call_vega'], 
                                    name='Call ν', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['put_vega'], 
                                    name='Put ν', line=dict(color='red')), row=2, col=1)
            
            # Theta
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['call_theta'], 
                                    name='Call Θ', line=dict(color='green')), row=2, col=2)
            fig.add_trace(go.Scatter(x=greeks_df['strike'], y=greeks_df['put_theta'], 
                                    name='Put Θ', line=dict(color='red')), row=2, col=2)
            
            # Add spot lines
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", row=i, col=j)
            
            fig.update_layout(height=800, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("📊 Market Interpretation")
            
            if total_delta > 10:
                delta_interpretation = "🟢 **BULLISH**: Total Delta > +10 indicates strong bullish positioning"
            elif total_delta < -10:
                delta_interpretation = "🔴 **BEARISH**: Total Delta < -10 indicates strong bearish positioning"
            else:
                delta_interpretation = "🟡 **SIDEWAYS**: Total Delta between -10 and +10 indicates neutral positioning"
            
            st.info(delta_interpretation)
            
            st.markdown("""
            **Greek Interpretations:**
            - **Delta**: Measures directional exposure. Positive = bullish, Negative = bearish
            - **Gamma**: Measures convexity. High gamma = potential for rapid moves
            - **Vega**: Measures volatility sensitivity. High vega = sensitive to IV changes
            - **Theta**: Measures time decay. Negative theta = losing value over time
            - **Rho**: Measures interest rate sensitivity
            """)
    
    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()

if __name__ == "__main__":
    main()