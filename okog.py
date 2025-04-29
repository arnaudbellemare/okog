# -*- coding: utf-8 -*-
"""
Paradex Options Snapshot Dashboard
"""

import streamlit as st
import datetime as dt
import scipy.stats as si
import scipy.interpolate
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.stats import linregress # For log-log regression
import statsmodels.tsa.stattools as smt # For autocovariance calculation
import pandas as pd
import requests
import numpy as np
import ccxt
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
from plotly.subplots import make_subplots
import math
import gc
import json # For inspecting API response
import plotly.graph_objects as go # Make sure this is imported at the top
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(layout="wide", page_title="Paradex Options Snapshot")

# --- Constants ---
PARADEX_MARKETS_URL = "https://api.testnet.paradex.trade/v1/markets"
REQUEST_TIMEOUT = 25 # Increased timeout

# --- Utility Functions ---

## Login Functions (Optional - Keep or Remove)
def load_credentials():
    """Load usernames and passwords from files."""
    try:
        with open("usernames.txt", "r") as f_user:
            users = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            pwds = [line.strip() for line in f_pass if line.strip()]
        if len(users) != len(pwds):
            st.error("Number of usernames and passwords mismatch.")
            return {}
        # st.success("Credentials loaded successfully.") # Can be noisy
        return dict(zip(users, pwds))
    except FileNotFoundError:
        # st.error("Credential files (usernames.txt, passwords.txt) not found.") # Only show error if login fails
        return {}
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    """Handle user login for the dashboard."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Please Log In")
        creds = load_credentials()
        # Dont stop here, allow viewing even if files missing, check on button click
        # if not creds:
        #     st.stop()

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if not creds:
                 st.error("Credential files not found or invalid.")
            elif username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully! Rerunning...")
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop() # Stop execution until logged in

## --- API Fetching ---

@st.cache_data(ttl=60) # Cache snapshot for 1 minute
def fetch_paradex_markets_snapshot(url=PARADEX_MARKETS_URL):
    """Fetches the current market snapshot from Paradex."""
    logging.info(f"Fetching Paradex markets snapshot from: {url}")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Check for HTTP errors
        logging.info(f"Successfully fetched Paradex data (Status: {response.status_code}).")
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching Paradex markets: {e}")
        logging.error(f"Network error fetching Paradex markets: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding Paradex API response (not valid JSON).")
        logging.error("Error decoding Paradex API response.")
        if response: logging.error("Response text:", response.text[:500]) # Print snippet to log
        return None
    except Exception as e:
        st.error(f"Error processing Paradex market data: {e}")
        logging.error(f"Error processing Paradex market data: {e}")
        return None

def filter_and_process_paradex_options(markets_data):
    """
    Filters the raw markets snapshot data for options and processes field names.
    *** REQUIRES ADJUSTMENT BASED ON ACTUAL API RESPONSE ***
    """
    if not markets_data:
        logging.warning("filter_and_process_paradex_options: Received empty markets_data.")
        return pd.DataFrame()

    market_list = []
    if isinstance(markets_data, dict):
        logging.info(f"Paradex API response keys: {list(markets_data.keys())}")
        data_key = 'results' # <<< --- ADJUST THIS KEY --- <<<
        if data_key in markets_data and isinstance(markets_data[data_key], list):
            market_list = markets_data[data_key]
            logging.info(f"Extracted market list from key '{data_key}'. Count: {len(market_list)}")
        else:
            logging.error(f"Could not find list of markets under expected key '{data_key}'.")
            return pd.DataFrame()
    elif isinstance(markets_data, list):
        market_list = markets_data
        logging.info(f"Paradex API response is a list. Count: {len(market_list)}")
    else:
        logging.error(f"Unexpected Paradex API response type: {type(markets_data)}")
        return pd.DataFrame()

    if not market_list:
        logging.warning("Paradex market list is empty.")
        return pd.DataFrame()

    options_processed = []
    logging.info("Filtering for options contracts and processing...")
    skipped_non_option = 0
    skipped_missing_essential = 0
    processed_count = 0

    for market in market_list:
        if not isinstance(market, dict): continue

        # --- Filtering based on fields ---
        asset_kind = market.get('asset_kind', '').upper()
        symbol = market.get('symbol', '') # <<<--- ADJUST KEY 'symbol' if needed
        option_type_field = market.get('option_type', '').upper() # <<<--- ADJUST KEY 'option_type' if needed

        # --- Determine if it's a standard option we want ---
        # Prioritize asset_kind if reliable, e.g., 'OPTION' vs 'PERP_OPTION'
        is_option = False
        if asset_kind == 'OPTION': # <<<--- ADJUST VALUE 'OPTION' if needed
             is_option = True
        elif option_type_field in ['CALL', 'PUT']: # Fallback to option_type field
             is_option = True
        elif len(symbol.split('-')) == 4 and symbol.split('-')[-1] in ['C', 'P']: # Fallback to symbol format
             # Be cautious with this one if expiry isn't in symbol
             is_option = True

        if not is_option:
            skipped_non_option += 1
            continue

        # --- Extract Essential Data ---
        # *** ADJUST KEYS AS NEEDED ***
        mark_price_key = 'mark_price'
        iv_key = 'iv'
        oi_key = 'open_interest'
        strike_key = 'strike_price'
        # *** CRITICAL: FIND THE EXPIRY TIMESTAMP FIELD ***
        expiry_key = 'expiry_at' # This was 0 in example, NEED THE CORRECT ONE

        # --- Validate and Process ---
        try:
            mark_price = market.get(mark_price_key)
            iv = market.get(iv_key)
            open_interest = market.get(oi_key)
            strike_price = market.get(strike_key)
            expiry_timestamp_ms = market.get(expiry_key) # Assume Milliseconds

            # Convert and validate essential numeric fields
            mark_price_f = float(mark_price) if mark_price is not None else np.nan
            iv_f = float(iv) if iv is not None else np.nan
            open_interest_f = float(open_interest) if open_interest is not None else np.nan
            strike_price_val = float(strike_price) if strike_price is not None else np.nan

            # Validate Expiry Timestamp
            if expiry_timestamp_ms is None or expiry_timestamp_ms <= 0:
                 logging.debug(f"Missing or invalid expiry timestamp ({expiry_timestamp_ms}) for {symbol}. Skipping.")
                 skipped_missing_essential += 1
                 continue
            else:
                 expiry_date_val = pd.to_datetime(expiry_timestamp_ms / 1000, unit='s', utc=True)

            # Check if essential numeric values are valid *after* conversion
            if pd.isna(mark_price_f) or pd.isna(iv_f) or pd.isna(open_interest_f) or pd.isna(strike_price_val):
                logging.debug(f"Missing essential numeric value for {symbol} after conversion. Skipping.")
                skipped_missing_essential += 1
                continue

            # Basic sanity check for IV (positive decimal)
            if not (0 < iv_f < 5.0):
                 logging.debug(f"Unusual IV value ({iv_f:.4f}) for {symbol}. Skipping.")
                 skipped_missing_essential += 1
                 continue

            # Derive option type ('C' or 'P') consistently
            option_type_derived = None
            if option_type_field == 'CALL': option_type_derived = 'C'
            elif option_type_field == 'PUT': option_type_derived = 'P'
            elif len(parts := symbol.split('-')) == 4 and parts[-1] in ['C', 'P']: option_type_derived = parts[-1] # Fallback
            else:
                 logging.warning(f"Could not derive option type for {symbol}. Skipping.")
                 skipped_missing_essential += 1
                 continue

            # Store Processed Data
            processed = {
                'instrument_name': symbol,
                'k': strike_price_val,
                'option_type': option_type_derived,
                'mark_price_close': mark_price_f,
                'open_interest': open_interest_f,
                'iv_close': iv_f,
                'expiry_date_utc': expiry_date_val,
                'underlying_asset': market.get('base_currency', '').upper() # <<<--- ADJUST KEY 'base_currency' if needed
            }

            # Add Greeks if they exist
            # *** ADJUST GREEK FIELD NAMES HERE in greek_map ***
            greek_map = {'delta': 'delta', 'gamma': 'gamma', 'vega': 'vega', 'theta': 'theta'}
            for dash_name, api_name in greek_map.items():
                 greek_val_raw = market.get(api_name)
                 try: processed[dash_name] = float(greek_val_raw) if greek_val_raw is not None else np.nan
                 except (ValueError, TypeError): processed[dash_name] = np.nan

            options_processed.append(processed)
            processed_count += 1

        except Exception as e:
            logging.warning(f"Error processing market {symbol}: {e}. Skipping.")
            skipped_missing_essential += 1
            continue

    logging.info(f"Processing Summary: Processed Options={processed_count}, Skipped (Non-Option)={skipped_non_option}, Skipped (Missing Essential)={skipped_missing_essential}")

    if not options_processed:
        logging.warning("No options found after filtering and processing.")
        return pd.DataFrame()

    df_options = pd.DataFrame(options_processed)
    logging.info(f"Final processed options DataFrame shape: {df_options.shape}")
    return df_options


## Spot Data Fetching (Keep Kraken for now)
@st.cache_data(ttl=60)
def fetch_kraken_data(coin="BTC", days=1): # Reduced days needed
    """Fetch recent 5-minute OHLCV data from Kraken."""
    logging.info(f"Fetching {days} days of 5m Kraken data for {coin}/USD.")
    try:
        k = ccxt.kraken()
        now_dt = dt.datetime.now(dt.timezone.utc) # Ensure UTC
        start_dt = now_dt - dt.timedelta(days=days)
        since = int(start_dt.timestamp() * 1000) # Milliseconds for CCXT
        symbol = f"{coin}/USD"
        ohlcv = k.fetch_ohlcv(symbol, timeframe="5m", since=since) # No limit

        if not ohlcv:
            st.warning(f"No recent 5m OHLCV data returned from Kraken for {symbol}.")
            return pd.DataFrame()

        dfr = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dfr["date_time"] = pd.to_datetime(dfr["timestamp"], unit="ms", errors='coerce').dt.tz_localize("UTC")
        dfr = dfr.dropna(subset=['date_time'])
        dfr.sort_values("date_time", inplace=True)
        # Don't filter by days again, just return what was fetched since 'since'
        dfr = dfr.reset_index(drop=True)
        logging.info(f"Kraken 5m data fetched. Shape: {dfr.shape}")
        return dfr

    except Exception as e:
        st.error(f"Error fetching Kraken 5m data: {e}")
        logging.error(f"Error fetching Kraken 5m data: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache daily data longer
def fetch_kraken_data_daily(days=365, coin="BTC"):
    """Fetch daily OHLCV data from Kraken."""
    # ... (fetch_kraken_data_daily code remains the same) ...
    logging.info(f"Fetching {days} days of daily Kraken data for {coin}/USD.")
    try:
        k = ccxt.kraken()
        now_dt = dt.datetime.now(dt.timezone.utc) # Ensure UTC
        start_dt = now_dt - dt.timedelta(days=days)
        since = int(start_dt.timestamp() * 1000)
        symbol = f"{coin}/USD"
        ohlcv = k.fetch_ohlcv(symbol, timeframe="1d", since=since)
        if not ohlcv:
            st.warning(f"No daily OHLCV data returned from Kraken for {symbol}.")
            return pd.DataFrame()

        dfr = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dfr["date_time"] = pd.to_datetime(dfr["timestamp"], unit="ms", errors='coerce').dt.tz_localize("UTC")
        dfr = dfr.dropna(subset=['date_time'])
        dfr.sort_values("date_time", inplace=True)
        dfr.reset_index(drop=True, inplace=True)
        logging.info(f"Kraken daily data fetched. Shape: {dfr.shape}")
        return dfr

    except Exception as e:
        st.error(f"Unexpected error fetching Kraken daily data: {e}")
        logging.error(f"Unexpected error fetching Kraken daily data: {e}")
        return pd.DataFrame()


## Expiry Option Parsing (Adapted for Paradex Snapshot)
def get_valid_expiration_options_from_snapshot(df_paradex_options):
    """Extracts valid future expiration dates from the processed snapshot DataFrame."""
    # ... (code remains the same) ...
    if df_paradex_options.empty or 'expiry_date_utc' not in df_paradex_options.columns:
        st.error("Cannot get expirations: Invalid or empty options snapshot DataFrame.")
        return []

    now_utc = dt.datetime.now(dt.timezone.utc)
    # Ensure the column is actually datetime objects before comparison
    df_paradex_options['expiry_date_utc'] = pd.to_datetime(df_paradex_options['expiry_date_utc'], errors='coerce')
    valid_dates = df_paradex_options['expiry_date_utc'].dropna().unique()

    # Filter for future dates
    future_dates = [d for d in valid_dates if pd.Timestamp(d) > now_utc] # Use pd.Timestamp for comparison

    if not future_dates:
        st.warning("No future expiration dates found in the current market snapshot.")

    return sorted(list(future_dates))


## Greeks Calculations (Keep, maybe needed if Paradex doesn't provide them)
# Ensure these use expiry_date_utc and handle potential NaNs
def compute_delta(row, S, snapshot_time_utc):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k')
        sigma = row.get('iv_close')
        option_type = row.get('option_type')
        expiry_date = row.get('expiry_date_utc') # Get pre-parsed expiry

        if pd.isna(k) or pd.isna(sigma) or pd.isna(S) or pd.isna(option_type) or pd.isna(expiry_date): return np.nan
        if S <= 0 or k <= 0: return np.nan
        if sigma < 1e-9: return 1.0 if option_type == 'C' and S > k else (-1.0 if option_type == 'P' and S < k else 0.0)

        T = (expiry_date - snapshot_time_utc).total_seconds() / (365 * 24 * 3600)
        if T < 1e-9: return 1.0 if option_type == 'C' and S > k else (-1.0 if option_type == 'P' and S < k else 0.0)

        sqrt_T = math.sqrt(T)
        denominator_d1 = sigma * sqrt_T
        if abs(denominator_d1) < 1e-12: return np.nan

        d1 = (np.log(S / k) + 0.5 * sigma**2 * T) / denominator_d1
        if not np.isfinite(d1): return np.nan

        delta_val = si.norm.cdf(d1) if option_type == 'C' else si.norm.cdf(d1) - 1.0
        return delta_val if np.isfinite(delta_val) else np.nan
    except Exception as e: logging.error(f"compute_delta ERROR ({instr_name}): {e}"); return np.nan

def compute_gamma(row, S, snapshot_time_utc):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close'); expiry_date = row.get('expiry_date_utc')
        if pd.isna(k) or pd.isna(sigma) or pd.isna(S) or pd.isna(expiry_date) or S <= 0 or sigma <= 0: return np.nan
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365 * 24 * 3600)
        if T <= 1e-9: return 0.0
        sqrt_T = math.sqrt(T); denominator_d1 = sigma * sqrt_T
        if denominator_d1 < 1e-9: return np.nan
        d1 = (math.log(S / k) + 0.5 * sigma**2 * T) / denominator_d1
        pdf_d1 = norm.pdf(d1)
        denominator_gamma = S * sigma * sqrt_T
        if denominator_gamma < 1e-9: return np.nan
        gamma_val = pdf_d1 / denominator_gamma
        return gamma_val if np.isfinite(gamma_val) else np.nan
    except Exception as e: logging.error(f"compute_gamma ERROR ({instr_name}): {e}"); return np.nan

def compute_vega(row, S, snapshot_time_utc):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close'); expiry_date = row.get('expiry_date_utc')
        if pd.isna(k) or pd.isna(sigma) or pd.isna(S) or pd.isna(expiry_date) or S <= 0: return np.nan
        if sigma <= 0: return 0.0
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365 * 24 * 3600)
        if T <= 1e-9: return 0.0
        sqrt_T = math.sqrt(T); denominator_d1 = sigma * sqrt_T
        if denominator_d1 < 1e-9: return np.nan # Avoid division by zero if sigma or T is tiny
        d1 = (math.log(S / k) + 0.5 * sigma**2 * T) / denominator_d1
        vega = S * norm.pdf(d1) * sqrt_T * 0.01 # Per 1% change in vol
        return vega if np.isfinite(vega) else np.nan
    except Exception as e: logging.error(f"compute_vega ERROR ({instr_name}): {e}"); return np.nan

def compute_charm(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close'); expiry_date = row.get('expiry_date_utc')
        if pd.isna(k) or pd.isna(sigma) or pd.isna(S) or pd.isna(expiry_date) or S <= 0 or k <= 0: return np.nan
        if sigma <= 1e-6: return 0.0
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365 * 24 * 3600)
        if T <= 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if sigma_sqrt_T < 1e-9: return 0.0
        b = r; log_moneyness = np.log(S / k)
        d1 = (log_moneyness + (b + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T; pdf_d1 = norm.pdf(d1)
        if not np.isfinite(d1) or not np.isfinite(d2) or not np.isfinite(pdf_d1): return np.nan
        charm_annual = -pdf_d1 * d2 / (2 * T) # Assuming b=r=0 for simplicity in crypto context
        charm_daily = charm_annual / 365.0
        return charm_daily if np.isfinite(charm_daily) else np.nan
    except Exception as e: logging.error(f"compute_charm ERROR ({instr_name}): {e}"); return np.nan

def compute_vanna(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close'); expiry_date = row.get('expiry_date_utc')
        if pd.isna(k) or pd.isna(sigma) or pd.isna(S) or pd.isna(expiry_date) or S <= 0 or k <= 0: return np.nan
        if sigma <= 1e-6: return 0.0
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365 * 24 * 3600)
        if T <= 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if sigma_sqrt_T < 1e-9: return 0.0
        log_moneyness = np.log(S / k)
        d1 = (log_moneyness + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T; pdf_d1 = norm.pdf(d1)
        if not np.isfinite(d1) or not np.isfinite(d2) or not np.isfinite(pdf_d1): return np.nan
        vanna = -math.exp(-r * T) * pdf_d1 * d2 / sigma # Standard def
        return vanna if np.isfinite(vanna) else np.nan
    except Exception as e: logging.error(f"compute_vanna ERROR ({instr_name}): {e}"); return np.nan

def compute_gex(row, S, oi): # Keep GEX helper
    # ... (compute_gex code remains the same) ...
    try:
        gamma_val = row.get('gamma')
        oi_val = float(oi) if pd.notna(oi) else np.nan
        if pd.isna(gamma_val) or pd.isna(oi_val) or pd.isna(S) or S <= 0 or oi_val < 0: return np.nan
        gex = gamma_val * oi_val * (S ** 2) * 0.01
        return gex if np.isfinite(gex) else np.nan
    except Exception as e: logging.error(f"compute_gex ERROR: {e}"); return np.nan


## Build Ticker List (Adapted for Paradex DataFrame)
def build_paradex_ticker_list(dft_latest_paradex):
    # ... (code remains the same) ...
    if dft_latest_paradex.empty: return []
    required_cols = ['instrument_name', 'k', 'option_type', 'iv_close', 'open_interest']
    greeks_present = [g for g in ['delta', 'gamma', 'vega', 'theta', 'charm', 'vanna'] if g in dft_latest_paradex.columns]
    if not all(c in dft_latest_paradex.columns for c in required_cols): return []

    ticker_list = []
    for _, row in dft_latest_paradex.iterrows():
        if pd.isna(row['k']) or pd.isna(row['iv_close']) or pd.isna(row['open_interest']): continue
        entry = {
            "instrument": row['instrument_name'], "strike": int(row['k']), "option_type": row['option_type'],
            "open_interest": float(row['open_interest']), "iv": float(row['iv_close']),
            **{greek: float(row[greek]) for greek in greeks_present if pd.notna(row[greek])}
        }
        ticker_list.append(entry)
    ticker_list.sort(key=lambda x: x['strike'])
    return ticker_list

## Plotting Functions (SNAPSHOT ONLY)
def plot_oi_by_strike(ticker_list, spot_price):
    """Plots Open Interest (Calls and Puts) by strike for the latest snapshot."""
    st.subheader("Open Interest by Strike (Latest Snapshot)")
    if not ticker_list:
        st.warning("Cannot plot OI by Strike: Ticker list is empty.")
        return

    try:
        df_ticker = pd.DataFrame(ticker_list)
        if df_ticker.empty or not all(c in df_ticker.columns for c in ['strike', 'open_interest', 'option_type']):
            st.warning("Cannot plot OI by Strike: Ticker list invalid or missing columns.")
            return

        # Aggregate OI per strike for calls and puts
        df_agg = df_ticker.pivot_table(index='strike', columns='option_type', values='open_interest', aggfunc='sum', fill_value=0)
        df_agg = df_agg.rename(columns={'C': 'call_oi', 'P': 'put_oi'}).reset_index()

        # Ensure columns exist even if one type had no OI
        if 'call_oi' not in df_agg.columns: df_agg['call_oi'] = 0
        if 'put_oi' not in df_agg.columns: df_agg['put_oi'] = 0

        df_agg['total_oi'] = df_agg['call_oi'] + df_agg['put_oi']
        df_agg = df_agg[df_agg['total_oi'] > 0] # Exclude strikes with zero total OI

        if df_agg.empty:
            st.warning("No Open Interest found for any strike in the latest snapshot.")
            return

        # Find strike with max total OI
        max_oi_idx = df_agg['total_oi'].idxmax()
        max_oi_strike = df_agg.loc[max_oi_idx, 'strike']
        max_oi_value = df_agg.loc[max_oi_idx, 'total_oi']

        # Plotting
        fig_oi = px.bar(
            df_agg,
            x='strike',
            y=['call_oi', 'put_oi'],
            title=f"Open Interest by Strike (Max OI at {max_oi_strike:,.0f})",
            labels={'value': 'Open Interest (Contracts)', 'strike': 'Strike Price', 'variable': 'Option Type'},
            color_discrete_map={'call_oi': 'mediumseagreen', 'put_oi': 'lightcoral'},
            barmode='group' # Group bars for calls and puts side-by-side
        )

        # Add Spot Price line
        fig_oi.add_vline(x=spot_price, line_dash="dot", line_color="grey", line_width=1.5,
                        annotation_text=f"Spot: {spot_price:,.0f}", annotation_position="top right")

        # Add Max OI Strike line and annotation
        fig_oi.add_vline(x=max_oi_strike, line_dash="dash", line_color="orange", line_width=2,
                        annotation_text=f"Max OI Strike: {max_oi_strike:,.0f}",
                        annotation_position="top left", annotation_font=dict(color="orange"))

        fig_oi.update_layout(height=400, width=800, bargap=0.1, legend_title_text=None)
        st.plotly_chart(fig_oi, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting OI by Strike: {e}")
        logging.error("Error in plot_oi_by_strike", exc_info=True)
def plot_open_interest_delta(ticker_list, spot_price):
    """Plot open interest and delta using pre-built ticker_list."""
    st.subheader("Open Interest & Delta Bubble Chart") # Keep the bubble chart part
    if not ticker_list:
        st.warning("No ticker data available for OI/Delta bubble chart.")
        return

    try:
        df_ticker = pd.DataFrame(ticker_list)
        if df_ticker.empty or not all(c in df_ticker.columns for c in ['strike', 'open_interest', 'delta', 'instrument', 'option_type', 'iv']): # Added option_type, iv
            st.warning("Ticker list is empty or missing required columns for bubble chart.")
            return

        df_plot = df_ticker.dropna(subset=['strike', 'open_interest', 'delta'])
        if df_plot.empty:
            st.warning("No valid data points for OI/Delta bubble chart.")
            return

        # Add moneyness for context
        df_plot['moneyness'] = df_plot['strike'] / spot_price

        # --- Bubble Chart (Keep as is) ---
        fig_bubble = px.scatter(
            df_plot,
            x="strike", y="delta", size="open_interest",
            color="moneyness", color_continuous_scale=px.colors.diverging.RdYlBu_r,
            range_color=[0.8, 1.2], hover_data=["instrument", "open_interest", "iv"],
            size_max=50, title=f"Open Interest & Delta by Strike (Size=OI, Color=Moneyness vs Spot={spot_price:.0f})"
        )
        fig_bubble.add_vline(x=spot_price, line_dash="dot", line_color="black", annotation_text="Spot")
        fig_bubble.add_hline(y=0.5, line_dash="dot", line_color="grey", line_width=1)
        fig_bubble.add_hline(y=-0.5, line_dash="dot", line_color="grey", line_width=1)
        fig_bubble.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_bubble.update_layout(height=500, width=900, xaxis_title="Strike Price", yaxis_title="Delta")
        st.plotly_chart(fig_bubble, use_container_width=True)
        # --- End Bubble Chart ---


        # --- OI Put/Call Ratio Gauge ---
        st.markdown("---") # Add separator
        st.subheader("OI Sentiment Gauge") # Add subheader for clarity

        # Ensure 'option_type' column exists before filtering
        if 'option_type' not in df_plot.columns:
            st.warning("Cannot calculate OI Ratio: 'option_type' column missing.")
            return

        total_oi = df_plot["open_interest"].sum()
        put_oi = df_plot[df_plot['option_type'] == 'P']['open_interest'].sum()
        call_oi = df_plot[df_plot['option_type'] == 'C']['open_interest'].sum() # Calculate Call OI too

        if total_oi > 0:
            # --- Calculate Put/Call Ratio (Standard Definition: Puts / Calls) ---
            # Avoid division by zero if call_oi is 0
            put_call_ratio = put_oi / call_oi if call_oi > 1e-9 else np.inf
            # Use ratio for interpretation, but plot the Put Percentage for the gauge
            put_percentage = (put_oi / total_oi) * 100

            # --- Determine Sentiment based on Put/Call Ratio ---
            if put_call_ratio == np.inf:
                 sentiment = "Extreme Bearish (Only Puts)"
                 gauge_color = "darkred"
            elif put_call_ratio > 0.7: # Common threshold, adjust as needed
                 sentiment = f"Bearish Lean (P/C Ratio: {put_call_ratio:.2f})"
                 gauge_color = "lightcoral"
            elif put_call_ratio < 0.5: # Common threshold, adjust as needed
                 sentiment = f"Bullish Lean (P/C Ratio: {put_call_ratio:.2f})"
                 gauge_color = "lightgreen"
            else: # Between 0.5 and 0.7
                 sentiment = f"Neutral (P/C Ratio: {put_call_ratio:.2f})"
                 gauge_color = "grey"

            st.markdown(f"##### OI Sentiment: {sentiment}") # Show interpretation

            # --- Corrected Gauge Plot ---
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", # Keep number display
                value = put_percentage, # Value being shown (0-100 scale)
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Put OI as % of Total OI", 'font': {'size': 16}}, # More descriptive title
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
                    'bar': {'color': gauge_color, 'thickness': 0.3}, # Make bar visible, set color based on sentiment
                    'bgcolor': "rgba(0,0,0,0.1)", # Slightly visible background
                    'borderwidth': 1,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(44, 160, 44, 0.4)'}, # Green background for < 50%
                        {'range': [50, 100], 'color': 'rgba(214, 39, 40, 0.4)'}], # Red background for > 50%
                    'threshold': {
                        'line': {'color': "white", 'width': 3}, # White threshold line at 50
                        'thickness': 0.75,
                        'value': 50
                     }
                }
            ))

            fig_gauge.update_layout(
                 height=250, width=350,
                 margin=dict(l=30, r=30, t=60, b=30), # Adjusted margins
                 paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                 font=dict(color="white") # Ensure text is visible on dark theme
                 )
            st.plotly_chart(fig_gauge, use_container_width=False) # Don't force width
            st.caption(f"Calculation: (Total Put OI / Total OI) * 100 = ({put_oi:,.0f} / {total_oi:,.0f}) * 100 = {put_percentage:.1f}%")


        else:
            st.warning("Total Open Interest is zero, cannot calculate ratio.")

    except Exception as e:
        st.error(f"Error plotting OI/Delta bubble chart or gauge: {e}")
        logging.error(f"Error plotting OI/Delta bubble chart or gauge: {e}", exc_info=True)
def plot_delta_balance(ticker_list, spot_price):
    """Plot the balance between put and call deltas weighted by open interest."""
    st.subheader("Put vs Call Delta Balance (OI Weighted)")
    if not ticker_list:
        st.warning("No ticker data available to calculate delta balance.")
        return

    try:
        # Calculate weighted deltas directly from the list
        call_weighted_delta = sum(item["delta"] * item["open_interest"] for item in ticker_list if item["option_type"] == "C" and not pd.isna(item["delta"]) and not pd.isna(item["open_interest"]))
        # Put delta is negative, use abs() for magnitude comparison if desired, but net calc needs sign
        put_delta_sum = sum(item["delta"] * item["open_interest"] for item in ticker_list if item["option_type"] == "P" and not pd.isna(item["delta"]) and not pd.isna(item["open_interest"]))

        delta_data = pd.DataFrame({
            'Option Type': ['Calls', 'Puts'],
            # Plot absolute values for easier comparison of magnitude
            'Total Weighted Delta': [call_weighted_delta, abs(put_delta_sum)],
            'Direction': ['Bullish Exposure', 'Bearish Exposure']
        })

        fig = px.bar(
            delta_data,
            x='Option Type',
            y='Total Weighted Delta',
            color='Direction',
            color_discrete_map={'Bullish Exposure': 'mediumseagreen', 'Bearish Exposure': 'lightcoral'},
            title='Put vs Call Delta Balance (Open Interest Weighted)',
            labels={'Total Weighted Delta': 'Absolute Total Delta * Open Interest'}
        )

        # Calculate Net Delta (Call Delta + Put Delta)
        net_delta = call_weighted_delta + put_delta_sum
        bias_strength = abs(net_delta)

        if net_delta > 0.01: # Add a small threshold to avoid near-zero noise
            bias_text = f"Market Bias: Bullish (Net Delta: +{net_delta:.2f})"
            bias_color = "green"
        elif net_delta < -0.01:
            bias_text = f"Market Bias: Bearish (Net Delta: {net_delta:.2f})"
            bias_color = "red"
        else:
            bias_text = f"Market Bias: Neutral (Net Delta: {net_delta:.2f})"
            bias_color = "grey"


        fig.add_annotation(
            x=0.5, y=1.05, xref="paper", yref="paper",
            text=bias_text, showarrow=False,
            font=dict(size=14, color=bias_color), align="center",
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=bias_color, borderwidth=1, borderpad=4
        )

        # Calculate Call/Put Ratio using absolute put delta sum for magnitude ratio
        abs_put_delta = abs(put_delta_sum)
        if abs_put_delta > 0:
            delta_ratio = call_weighted_delta / abs_put_delta
            fig.add_annotation(
                x=0.5, y=-0.15, xref="paper", yref="paper", # Position below x-axis
                text=f"Call/Put Delta Ratio (Magnitude): {delta_ratio:.2f}",
                showarrow=False, font=dict(size=12), align="center"
            )

        # Prepare data for strike range plot
        call_items = [item for item in ticker_list if item["option_type"] == "C"]
        put_items = [item for item in ticker_list if item["option_type"] == "P"]

        fig_strikes = create_delta_by_strike_chart(call_items, put_items, spot_price)

        st.plotly_chart(fig, use_container_width=True)
        if fig_strikes:
            st.plotly_chart(fig_strikes, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting Delta Balance: {e}")
        logging.error(f"Error plotting Delta Balance: {e}")


def create_delta_by_strike_chart(call_items, put_items, spot_price):
    """Create a chart of delta distribution by strike range (relative to spot)."""
    # Define strike ranges relative to spot price
    ranges = [
        {"name": "Deep ITM (< 0.9 S)", "min_pct": -float('inf'), "max_pct": -0.10}, # Puts deep ITM / Calls deep OTM
        {"name": "ITM (0.9 - 0.98 S)", "min_pct": -0.10, "max_pct": -0.02},        # Puts ITM / Calls OTM
        {"name": "Near ATM (0.98 - 1.02 S)", "min_pct": -0.02, "max_pct": 0.02},   # ATM
        {"name": "OTM (1.02 - 1.1 S)", "min_pct": 0.02, "max_pct": 0.10},          # Puts OTM / Calls ITM
        {"name": "Deep OTM (> 1.1 S)", "min_pct": 0.10, "max_pct": float('inf')}   # Puts deep OTM / Calls deep ITM
    ]
    range_data = []

    # Process Calls
    for strike_range in ranges:
        min_strike = spot_price * (1 + strike_range["min_pct"])
        max_strike = spot_price * (1 + strike_range["max_pct"])
        calls_in_range = [
            item for item in call_items
            if min_strike <= item["strike"] < max_strike and not pd.isna(item["delta"]) and not pd.isna(item["open_interest"])
        ]
        call_delta = sum(item["delta"] * item["open_interest"] for item in calls_in_range)
        range_data.append({
            "Strike Range": strike_range["name"],
            "Option Type": "Calls",
            "Weighted Delta": call_delta,
            "Sort Order": ranges.index(strike_range) # Keep original sort order
        })

    # Process Puts
    for strike_range in ranges:
        min_strike = spot_price * (1 + strike_range["min_pct"])
        max_strike = spot_price * (1 + strike_range["max_pct"])
        puts_in_range = [
            item for item in put_items
            if min_strike <= item["strike"] < max_strike and not pd.isna(item["delta"]) and not pd.isna(item["open_interest"])
        ]
        # Sum the actual put delta (negative values)
        put_delta = sum(item["delta"] * item["open_interest"] for item in puts_in_range)
        range_data.append({
            "Strike Range": strike_range["name"],
            "Option Type": "Puts",
            "Weighted Delta": put_delta, # Keep the sign
            "Sort Order": ranges.index(strike_range)
        })

    if not range_data: return None

    try:
        df_range = pd.DataFrame(range_data)
        df_range = df_range.sort_values("Sort Order") # Sort by the defined order

        fig = px.bar(
            df_range,
            x="Strike Range",
            y="Weighted Delta",
            color="Option Type",
            barmode="group",
            title="Delta Exposure Distribution by Strike Range (Relative to Spot)",
            color_discrete_map={"Calls": "mediumseagreen", "Puts": "lightcoral"},
            labels={"Weighted Delta": "Total Delta * OI", "Strike Range": "Strike Range vs Spot Price"},
            category_orders={"Strike Range": [r["name"] for r in ranges]} # Ensure correct x-axis order
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(height=400, width=800)
        return fig
    except Exception as e:
        st.error(f"Error creating delta by strike chart: {e}")
        logging.error(f"Error creating delta by strike chart: {e}")
        return None
def plot_premium_by_strike(dft_latest, spot_price):
    """Plots OI-Weighted Premium (Calls and Puts) by strike for the latest snapshot."""
    st.subheader("OI-Weighted Premium by Strike (Latest Snapshot)")

    required_cols = ['k', 'mark_price_close', 'open_interest', 'option_type']
    if dft_latest.empty or not all(c in dft_latest.columns for c in required_cols):
        st.warning("Cannot plot Premium by Strike: Latest data invalid or missing columns.")
        return

    try:
        df_calc = dft_latest.copy()
        df_calc['mark_price_close'] = pd.to_numeric(df_calc['mark_price_close'], errors='coerce')
        df_calc['open_interest'] = pd.to_numeric(df_calc['open_interest'], errors='coerce')

        # Calculate premium value = mark_price * OI
        df_calc['premium_value'] = df_calc['mark_price_close'].fillna(0) * df_calc['open_interest'].fillna(0)
        df_calc = df_calc.dropna(subset=['k', 'premium_value']) # Drop if strike or calculated premium is NaN

        # Aggregate premium per strike for calls and puts
        df_agg = df_calc.pivot_table(index='k', columns='option_type', values='premium_value', aggfunc='sum', fill_value=0)
        df_agg = df_agg.rename(columns={'C': 'call_prem', 'P': 'put_prem', 'k':'strike'}).reset_index()
        df_agg = df_agg.rename(columns={'k':'strike'}) # Ensure strike column name is consistent

        # Ensure columns exist even if one type had no premium
        if 'call_prem' not in df_agg.columns: df_agg['call_prem'] = 0
        if 'put_prem' not in df_agg.columns: df_agg['put_prem'] = 0

        df_agg['total_prem'] = df_agg['call_prem'] + df_agg['put_prem']
        df_agg = df_agg[df_agg['total_prem'] > 0] # Exclude strikes with zero total premium

        if df_agg.empty:
            st.warning("No Premium value found for any strike in the latest snapshot.")
            return

        # Find strike with max total premium
        max_prem_idx = df_agg['total_prem'].idxmax()
        max_prem_strike = df_agg.loc[max_prem_idx, 'strike']
        max_prem_value = df_agg.loc[max_prem_idx, 'total_prem']

        # Plotting
        fig_prem = px.bar(
            df_agg,
            x='strike',
            y=['call_prem', 'put_prem'],
            title=f"OI-Weighted Premium by Strike (Max Premium at {max_prem_strike:,.0f})",
            labels={'value': 'Total Premium ($ Value)', 'strike': 'Strike Price', 'variable': 'Option Type'},
            color_discrete_map={'call_prem': 'mediumseagreen', 'put_prem': 'lightcoral'},
            barmode='group'
        )

        # Add Spot Price line
        fig_prem.add_vline(x=spot_price, line_dash="dot", line_color="grey", line_width=1.5,
                         annotation_text=f"Spot: {spot_price:,.0f}", annotation_position="top right")

        # Add Max Premium Strike line and annotation
        fig_prem.add_vline(x=max_prem_strike, line_dash="dash", line_color="yellow", line_width=2,
                         annotation_text=f"Max Prem Strike: {max_prem_strike:,.0f}",
                         annotation_position="top left", annotation_font=dict(color="yellow"))

        fig_prem.update_layout(height=400, width=800, bargap=0.1, legend_title_text=None)
        st.plotly_chart(fig_prem, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting Premium by Strike: {e}")
        logging.error("Error in plot_premium_by_strike", exc_info=True)        
def plot_net_delta(df_ticker_list, spot_price=None): # Added optional spot_price for vline
    """Plot net delta exposure (Delta * OI) by strike."""
    st.subheader("Net Delta Exposure by Strike (Latest Snapshot)")

    # --- Input Validation ---
    if not isinstance(df_ticker_list, pd.DataFrame) or df_ticker_list.empty:
        st.warning("Net Delta: Input DataFrame is empty or not a DataFrame.")
        return
    required_cols = ['strike', 'delta', 'open_interest', 'option_type']
    if not all(c in df_ticker_list.columns for c in required_cols):
        st.warning(f"Net Delta: Input missing required columns ({required_cols}). Has: {df_ticker_list.columns.tolist()}")
        return

    df_plot_agg = df_ticker_list.dropna(subset=required_cols).copy() # Work on a copy
    if df_plot_agg.empty:
        st.warning("Net Delta: No valid data points after removing NaNs before aggregation.")
        return

    # --- Calculation ---
    try:
        # Calculate Delta * Open Interest for each option
        df_plot_agg['delta_oi'] = df_plot_agg['delta'] * df_plot_agg['open_interest']

        # Group by strike and sum the delta_oi for calls and puts separately, then net them out
        dfn = df_plot_agg.groupby("strike").apply(
            lambda x: x.loc[x["option_type"]=="C", "delta_oi"].sum(skipna=True) + x.loc[x["option_type"]=="P", "delta_oi"].sum(skipna=True),
            # Note: We ADD put delta_oi because put delta itself is negative.
            include_groups=False # pandas >= 1.5.0 ? check your version
        ).reset_index(name="net_delta_oi") # net_delta_oi = (Call Delta * OI) + (Put Delta * OI)

        if dfn.empty or dfn['net_delta_oi'].isna().all():
            st.warning("Net Delta: Aggregation result empty or all NaN.")
            return

    except Exception as e:
        st.error(f"Error during Net Delta aggregation: {e}")
        logging.error("Error during Net Delta aggregation", exc_info=True)
        return

    # --- Plotting ---
    try:
        dfn["sign"] = dfn["net_delta_oi"].apply(lambda v: "Negative Exposure" if v < 0 else "Positive Exposure")
        fig = px.bar(
            dfn, x="strike", y="net_delta_oi", color="sign",
            color_discrete_map={"Negative Exposure": "lightcoral", "Positive Exposure": "mediumseagreen"},
            title="Net Delta Exposure (Call Δ*OI + Put Δ*OI) (Latest Snapshot)",
            labels={"net_delta_oi": "Net Delta * OI Value", "strike": "Strike Price"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

        # Add spot price line if provided
        if spot_price is not None and pd.notna(spot_price):
            fig.add_vline(x=spot_price, line_dash="dot", line_color="grey", line_width=1,
                          annotation_text=f"Spot {spot_price:.0f}", annotation_position="top right", annotation_font_size=10)

        fig.update_layout(height=400, width=800, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting Net Delta: {e}")
        logging.error("Error plotting Net Delta", exc_info=True)
def plot_gex_by_strike(df_gex_input):
    """Plot gamma exposure (GEX) by strike."""
    st.subheader("Gamma Exposure (GEX) by Strike (Latest Snapshot)")
    if df_gex_input.empty: st.warning("GEX by Strike: Input DataFrame is empty."); return
    required_cols = ['k', 'gex', 'option_type']
    if not all(c in df_gex_input.columns for c in required_cols): st.warning(f"GEX by Strike: Input missing required columns ({required_cols})."); return

    df_gex = df_gex_input.copy()
    if 'strike' not in df_gex.columns and 'k' in df_gex.columns: df_gex['strike'] = df_gex['k']
    required_cols_plot = ['strike', 'gex', 'option_type']
    df_plot = df_gex.dropna(subset=required_cols_plot)

    if df_plot.empty: st.warning("GEX by Strike: No valid data points after removing NaNs from strike/gex/option_type."); return
    try:
        fig = px.bar(
            df_plot, x="strike", y="gex", color="option_type",
            title="Gamma Exposure (GEX) by Strike (Latest Snapshot)",
            labels={"gex": "GEX Value", "strike": "Strike Price", "option_type": "Type"},
            color_discrete_map={'C': 'mediumseagreen', 'P': 'lightcoral'}, barmode='group'
        )
        fig.update_layout(height=400, width=800, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error plotting GEX by strike: {e}"); logging.error("Error plotting GEX by strike", exc_info=True)

def plot_net_gex(df_gex_input, spot_price):
    """Plot net gamma exposure by strike."""
    st.subheader("Net Gamma Exposure by Strike (Latest Snapshot)")
    if df_gex_input.empty: st.warning("Net GEX: Input DataFrame is empty."); return
    required_cols = ['k', 'gex', 'option_type']
    if not all(c in df_gex_input.columns for c in required_cols): st.warning(f"Net GEX: Input missing required columns ({required_cols})."); return

    df_gex = df_gex_input.copy()
    if 'strike' not in df_gex.columns and 'k' in df_gex.columns: df_gex['strike'] = df_gex['k']
    required_cols_agg = ['strike', 'gex', 'option_type']
    df_plot_agg = df_gex.dropna(subset=required_cols_agg)

    if df_plot_agg.empty: st.warning("Net GEX: No valid data points after removing NaNs before aggregation."); return
    try:
        dfn = df_plot_agg.groupby("strike").apply(
            lambda x: x.loc[x["option_type"]=="C", "gex"].sum(skipna=True) - x.loc[x["option_type"]=="P", "gex"].sum(skipna=True),
            include_groups=False
        ).reset_index(name="net_gex")
        if dfn.empty or dfn['net_gex'].isna().all(): st.warning("Net GEX: Aggregation result empty or all NaN."); return
    except Exception as e: st.error(f"Error during Net GEX aggregation: {e}"); logging.error("Error during Net GEX aggregation", exc_info=True); return

    try:
        dfn["sign"] = dfn["net_gex"].apply(lambda v: "Negative" if v < 0 else "Positive")
        fig = px.bar(
            dfn, x="strike", y="net_gex", color="sign",
            color_discrete_map={"Negative": "orange", "Positive": "royalblue"},
            title="Net Gamma Exposure (Calls GEX - Puts GEX) (Latest Snapshot)",
            labels={"net_gex": "Net GEX Value", "strike": "Strike Price"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        if pd.notna(spot_price):
            fig.add_vline(x=spot_price, line_dash="dot", line_color="grey", line_width=1,
                          annotation_text=f"Spot {spot_price:.0f}", annotation_position="top right", annotation_font_size=10)
        fig.update_layout(height=400, width=800, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error plotting Net GEX: {e}"); logging.error("Error plotting Net GEX", exc_info=True)

def plot_call_put_max_pain(df_calls, df_puts, spot_price):
    """
    Plot continuous max pain curves, showing Total Max Pain and the
    prices where individual Call and Put writers' pain are minimized. <-- CLARIFIED
    """
    st.subheader("Max Pain Analysis (Continuous Price Range)")
    if df_calls.empty or df_puts.empty or 'open_interest' not in df_calls.columns:
        st.warning("Cannot compute Max Pain due to missing data or OI column.")
        return

    try:
        # --- OI Aggregation ---
        calls_agg = df_calls.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "call_oi"})
        puts_agg = df_puts.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "put_oi"})
        df_oi = pd.merge(calls_agg, puts_agg, on="strike", how="outer").fillna(0)
        df_oi = df_oi.sort_values("strike").reset_index(drop=True)
        if df_oi.empty:
            st.warning("No aggregated OI data for Max Pain.")
            return

        # --- Price Range ---
        # Use min/max strikes with OI to define calculation bounds relative to spot
        min_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].min()
        max_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].max()
        # Adjust range slightly based on spot and OI distribution
        low_price = max(0, min(spot_price * 0.8, min_strike_oi * 0.95))
        high_price = max(spot_price * 1.2, max_strike_oi * 1.05) # Ensure range covers max strike
        price_range = np.linspace(low_price, high_price, 150) # More points for smoothness

        # --- Pain Calculation ---
        results = []
        for price in price_range:
            call_losses = (price - df_oi["strike"]) * df_oi["call_oi"]
            call_pain = call_losses[call_losses > 0].sum()
            put_losses = (df_oi["strike"] - price) * df_oi["put_oi"]
            put_pain = put_losses[put_losses > 0].sum()
            total_pain = call_pain + put_pain
            results.append((price, call_pain, put_pain, total_pain))

        df_pain = pd.DataFrame(results, columns=["price", "call_pain", "put_pain", "total_pain"])

        # --- Find Minimum Pain Points ---
        # Price where TOTAL pain is minimized (Standard Max Pain)
        total_mp_idx = df_pain["total_pain"].idxmin()
        total_mp_price = df_pain.loc[total_mp_idx, "price"]

        # Price where CALL pain is minimized
        call_mp_idx = df_pain["call_pain"].idxmin()
        call_mp_price = df_pain.loc[call_mp_idx, "price"]

        # Price where PUT pain is minimized
        put_mp_idx = df_pain["put_pain"].idxmin()
        put_mp_price = df_pain.loc[put_mp_idx, "price"]
        # --- End Find Minimums ---

        # --- Plotting ---
        fig = go.Figure()
        # Plot individual pain curves
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["call_pain"], mode="lines", name="Call Writers' Pain", line=dict(color="green", width=1.5)))
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["put_pain"], mode="lines", name="Put Writers' Pain", line=dict(color="red", width=1.5)))
        # Plot total pain curve more prominently
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["total_pain"], mode="lines", name="Total Pain", line=dict(color="blue", width=3)))

        # --- Add Vertical Lines with Clearer Annotations ---
        # Line for minimum CALL pain
        fig.add_vline(x=call_mp_price, line_color="green", line_dash="dashdot", line_width=1,
                      annotation_text=f"Min Call Pain Price: {call_mp_price:.0f}",
                      annotation_position="top left")

        # Line for minimum PUT pain
        fig.add_vline(x=put_mp_price, line_color="red", line_dash="dashdot", line_width=1,
                      annotation_text=f"Min Put Pain Price: {put_mp_price:.0f}",
                      annotation_position="top right")

        # Line for minimum TOTAL pain (The actual Max Pain)
        fig.add_vline(x=total_mp_price, line_color="blue", line_dash="dash", line_width=2,
                      annotation_text=f"Total Max Pain Price: {total_mp_price:.0f}",
                      annotation_position="bottom")

        # Line for current Spot price
        fig.add_vline(x=spot_price, line_color="black", line_dash="dot", line_width=1.5,
                      annotation_text=f"Spot: {spot_price:.0f}",
                      annotation_position="bottom right") # Adjusted position

        fig.update_layout(title="Option Writers' Max Pain (Continuous Price Range)", # Kept title general
                          xaxis_title="Underlying Price at Expiry ($)",
                          yaxis_title="Aggregate Notional Loss ($)", # More descriptive Y-axis
                          legend_title="Pain Curves", width=900, height=500,
                          hovermode='x unified') # Better hover
        st.plotly_chart(fig, use_container_width=True)

        # --- Update the display text for clarity ---
        st.write(f"**Total Max Pain Price:** {total_mp_price:.0f} (Price minimizing combined writer losses)")
        st.write(f"**Min Call Pain Price:** {call_mp_price:.0f} (Price minimizing only call writer losses)")
        st.write(f"**Min Put Pain Price:** {put_mp_price:.0f} (Price minimizing only put writer losses)")
        st.caption("Note: The 'Total Max Pain Price' is the standard metric.")

    except Exception as e:
        st.error(f"Error plotting continuous Max Pain: {e}")
        logging.error(f"Error plotting continuous Max Pain: {e}", exc_info=True)

    st.subheader("Max Pain Analysis (Latest OI - Continuous Price)")
    required_cols_mp = ['k', 'option_type', 'open_interest']
    if dft_latest.empty or not all(c in dft_latest.columns for c in required_cols_mp):
        st.warning("Cannot compute Max Pain: Missing latest data or OI column."); return
    df_calls = dft_latest[dft_latest['option_type']=='C']; df_puts = dft_latest[dft_latest['option_type']=='P']
    try:
        calls_agg = df_calls.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "call_oi"})
        puts_agg = df_puts.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "put_oi"})
        df_oi = pd.merge(calls_agg, puts_agg, on="strike", how="outer").fillna(0); df_oi = df_oi.sort_values("strike").reset_index(drop=True)
        if df_oi.empty: st.warning("No aggregated OI data for Max Pain."); return
        min_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].min(); max_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].max()
        low_price = max(0, min(spot_price * 0.8, min_strike_oi * 0.95)); high_price = max(spot_price * 1.2, max_strike_oi * 1.05)
        price_range = np.linspace(low_price, high_price, 150)
        results = []
        for price in price_range:
            call_losses = (price - df_oi["strike"]) * df_oi["call_oi"]; call_pain = call_losses[call_losses > 0].sum()
            put_losses = (df_oi["strike"] - price) * df_oi["put_oi"]; put_pain = put_losses[put_losses > 0].sum()
            results.append((price, call_pain, put_pain, call_pain + put_pain))
        df_pain = pd.DataFrame(results, columns=["price", "call_pain", "put_pain", "total_pain"])
        total_mp_idx = df_pain["total_pain"].idxmin(); total_mp_price = df_pain.loc[total_mp_idx, "price"]
        call_mp_idx = df_pain["call_pain"].idxmin(); call_mp_price = df_pain.loc[call_mp_idx, "price"]
        put_mp_idx = df_pain["put_pain"].idxmin(); put_mp_price = df_pain.loc[put_mp_idx, "price"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["call_pain"], mode="lines", name="Call Writers' Pain", line=dict(color="green", width=1.5)))
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["put_pain"], mode="lines", name="Put Writers' Pain", line=dict(color="red", width=1.5)))
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["total_pain"], mode="lines", name="Total Pain", line=dict(color="blue", width=3)))
        fig.add_vline(x=call_mp_price, line_color="green", line_dash="dashdot", line_width=1, annotation_text=f"Min Call Pain: {call_mp_price:.0f}"); fig.add_vline(x=put_mp_price, line_color="red", line_dash="dashdot", line_width=1, annotation_text=f"Min Put Pain: {put_mp_price:.0f}")
        fig.add_vline(x=total_mp_price, line_color="blue", line_dash="dash", line_width=2, annotation_text=f"Total Max Pain: {total_mp_price:.0f}", annotation_position="bottom")
        fig.add_vline(x=spot_price, line_color="black", line_dash="dot", line_width=1.5, annotation_text=f"Spot: {spot_price:.0f}", annotation_position="bottom right")
        fig.update_layout(title="Option Writers' Max Pain (Latest OI)", xaxis_title="Underlying Price at Expiry ($)", yaxis_title="Aggregate Notional Loss ($)", legend_title="Pain Curves", width=900, height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Total Max Pain Price:** {total_mp_price:.0f}"); st.write(f"**Min Call Pain Price:** {call_mp_price:.0f}"); st.write(f"**Min Put Pain Price:** {put_mp_price:.0f}")
    except Exception as e: st.error(f"Error plotting Max Pain: {e}"); logging.error(f"Error plotting Max Pain: {e}", exc_info=True)

def plot_call_put_max_pain_in_range(dft_latest, spot_price, lower_factor=0.9, upper_factor=1.1):
    # ... (code adapted to take dft_latest) ...
    st.subheader(f"Call vs Put Pain Analysis (Latest OI - Zoomed Range: [{lower_factor:.1f}x - {upper_factor:.1f}x])")
    required_cols_mp = ['k', 'option_type', 'open_interest']
    if dft_latest.empty or not all(c in dft_latest.columns for c in required_cols_mp): st.warning("Cannot compute Zoomed Max Pain."); return
    df_calls = dft_latest[dft_latest['option_type']=='C']; df_puts = dft_latest[dft_latest['option_type']=='P']
    try:
        calls_agg = df_calls.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "call_oi"})
        puts_agg = df_puts.groupby("k")["open_interest"].sum().reset_index().rename(columns={"k": "strike", "open_interest": "put_oi"})
        df_oi = pd.merge(calls_agg, puts_agg, on="strike", how="outer").fillna(0); df_oi = df_oi.sort_values("strike").reset_index(drop=True)
        if df_oi.empty: st.warning("No aggregated OI data for Max Pain calculation."); return
        min_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].min(); max_strike_oi = df_oi[df_oi['call_oi'] + df_oi['put_oi'] > 0]['strike'].max()
        calc_low_price = max(0, min(spot_price * 0.7, min_strike_oi * 0.9)); calc_high_price = max(spot_price * 1.3, max_strike_oi * 1.1)
        price_range_calc = np.linspace(calc_low_price, calc_high_price, 200)
        results = []
        for price in price_range_calc:
            call_losses = (price - df_oi["strike"]) * df_oi["call_oi"]; call_pain = call_losses[call_losses > 0].sum()
            put_losses = (df_oi["strike"] - price) * df_oi["put_oi"]; put_pain = put_losses[put_losses > 0].sum()
            results.append((price, call_pain, put_pain, call_pain + put_pain))
        df_pain = pd.DataFrame(results, columns=["price", "call_pain", "put_pain", "total_pain"])
        if df_pain.empty: st.warning("Pain calculation resulted in empty DataFrame."); return
        df_pain['pain_diff'] = abs(df_pain['call_pain'] - df_pain['put_pain']); cross_idx = df_pain['pain_diff'].idxmin(); cross_price = df_pain.loc[cross_idx, "price"]
        total_mp_idx = df_pain["total_pain"].idxmin(); total_mp_price = df_pain.loc[total_mp_idx, "price"]
        zoom_low_price = lower_factor * spot_price; zoom_high_price = upper_factor * spot_price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["call_pain"], mode="lines", name="Call Writers' Pain", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["put_pain"], mode="lines", name="Put Writers' Pain", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=df_pain["price"], y=df_pain["total_pain"], mode="lines", name="Total Pain", line=dict(color="grey", dash='dot', width=1)))
        fig.add_vline(x=cross_price, line_color="purple", line_dash="dash", line_width=2, annotation_text=f"Crossover: {cross_price:.0f}", annotation_position="bottom right")
        fig.add_vline(x=spot_price, line_color="black", line_dash="dot", annotation_text=f"Spot: {spot_price:.0f}", annotation_position="top right")
        fig.update_xaxes(range=[zoom_low_price, zoom_high_price])
        df_pain_zoomed = df_pain[(df_pain['price'] >= zoom_low_price) & (df_pain['price'] <= zoom_high_price)]
        if not df_pain_zoomed.empty: min_y_zoom = 0; max_y_zoom = df_pain_zoomed[['call_pain', 'put_pain', 'total_pain']].max().max() * 1.1; fig.update_yaxes(range=[min_y_zoom, max_y_zoom])
        fig.update_layout(title="Call vs Put Writers' Pain (Zoomed View)", xaxis_title="Strike Price", yaxis_title="Total $ Loss (OI-Weighted)", legend_title="Pain Metrics", width=900, height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Price where Call Pain ≈ Put Pain: {cross_price:.0f}** (Total Max Pain: {total_mp_price:.0f})")
    except Exception as e: st.error(f"Error plotting zoomed Max Pain: {e}"); logging.error(f"Error plotting zoomed Max Pain: {e}", exc_info=True)

def plot_volatility_smile(dft_latest, spot_price):
     # ... (code remains the same, takes dft_latest) ...
    st.subheader("Volatility Smile (Latest Snapshot)")
    required_cols_smile = ['k', 'iv_close', 'option_type', 'date_time'] # Need type for OTM filtering, date_time for title
    if dft_latest.empty or not all(c in dft_latest.columns for c in required_cols_smile):
        st.warning("Volatility smile cannot be generated due to missing data.")
        return
    smile_df = dft_latest.dropna(subset=['k', 'iv_close']) # Drop only needed cols
    if smile_df.empty: st.warning("No valid data points for volatility smile."); return
    try:
        atm_option = smile_df.loc[abs(smile_df['k'] - spot_price).idxmin()]; atm_strike = atm_option['k']; atm_iv = atm_option['iv_close']
        smile_df = smile_df.sort_values(by="k")
        title_ts = smile_df['date_time'].iloc[0].strftime('%d %b %H:%M') if 'date_time' in smile_df.columns else 'Latest'
        fig_vol_smile = px.line(smile_df, x="k", y="iv_close", markers=True, title=f"Volatility Smile at {title_ts}", labels={"iv_close": "Implied Volatility", "k": "Strike Price"})
        fig_vol_smile.add_trace(go.Scatter(x=[atm_strike], y=[atm_iv], mode='markers', marker=dict(size=10, color='red', symbol='x'), name='ATM'))
        # Find cheapest hedge (simplistic)
        otm_puts = smile_df[(smile_df['k'] < spot_price) & (smile_df['option_type'] == 'P')]; otm_calls = smile_df[(smile_df['k'] > spot_price) & (smile_df['option_type'] == 'C')]
        cheapest_put_iv = otm_puts['iv_close'].min() if not otm_puts.empty else np.inf; cheapest_call_iv = otm_calls['iv_close'].min() if not otm_calls.empty else np.inf
        if cheapest_put_iv < cheapest_call_iv and not pd.isna(cheapest_put_iv): cheap_hedge_strike = otm_puts.loc[otm_puts['iv_close'].idxmin(), 'k']; annotation_text = f"Lowest IV OTM Put ({cheap_hedge_strike})"
        elif not pd.isna(cheapest_call_iv): cheap_hedge_strike = otm_calls.loc[otm_calls['iv_close'].idxmin(), 'k']; annotation_text = f"Lowest IV OTM Call ({cheap_hedge_strike})"
        else: cheap_hedge_strike = None
        if cheap_hedge_strike: fig_vol_smile.add_vline(x=cheap_hedge_strike, line=dict(dash="dash", color="green", width=1), annotation_text=annotation_text, annotation_position="top")
        fig_vol_smile.add_vline(x=spot_price, line=dict(dash="dot", color="black", width=1), annotation_text=f"Spot: {spot_price:.2f}", annotation_position="bottom right")
        fig_vol_smile.update_layout(height=400, width=700, yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol_smile, use_container_width=True)
    except Exception as e: st.error(f"Error plotting Volatility Smile: {e}"); logging.error(f"Error plotting Volatility Smile: {e}")

def compute_and_display_otm_average_skew(dft_latest, spot_price):
    """
    Computes and displays a skew metric based on the difference between
    the average IV of OTM calls and the average IV of OTM puts,
    normalized by the ATM IV.

    Skew ≈ [ Mean(σ_otm_call) - Mean(σ_otm_put) ] / σ_ATM
    """
    st.subheader("OTM Average Implied Volatility Skew (Normalized)")

    # --- 1. Input Checks ---
    required_cols = ['k', 'iv_close', 'option_type']
    if dft_latest.empty:
         st.warning("Cannot compute OTM Avg Skew: Latest options data ('dft_latest') is empty.")
         return
    if not all(c in dft_latest.columns for c in required_cols):
        missing = [c for c in required_cols if c not in dft_latest.columns]
        st.warning(f"Cannot compute OTM Avg Skew: Latest options data missing required columns: {', '.join(missing)}.")
        return
    if pd.isna(spot_price) or spot_price <= 0:
         st.warning(f"Cannot compute OTM Avg Skew: Invalid spot price ({spot_price}).")
         return

    # --- 2. Prepare Data & Calculate Average IVs ---
    # Use .copy() to avoid potential SettingWithCopyWarning issues later
    otm_calls = dft_latest[(dft_latest['option_type'] == 'C') & (dft_latest['k'] > spot_price)].copy()
    otm_puts = dft_latest[(dft_latest['option_type'] == 'P') & (dft_latest['k'] < spot_price)].copy()

    # Calculate Mean OTM Call IV
    mean_iv_otm_call = np.nan
    otm_calls_clean = otm_calls.dropna(subset=['iv_close'])
    otm_calls_clean = otm_calls_clean[otm_calls_clean['iv_close'] > 1e-6] # Ensure positive IV
    if not otm_calls_clean.empty:
        mean_iv_otm_call = otm_calls_clean['iv_close'].mean()
        logging.info(f"OTM Avg Skew: Calculated Mean OTM Call IV: {mean_iv_otm_call:.4f} from {len(otm_calls_clean)} options.")
    else:
        logging.warning("OTM Avg Skew: No valid OTM calls found to calculate average IV.")

    # Calculate Mean OTM Put IV
    mean_iv_otm_put = np.nan
    otm_puts_clean = otm_puts.dropna(subset=['iv_close'])
    otm_puts_clean = otm_puts_clean[otm_puts_clean['iv_close'] > 1e-6] # Ensure positive IV
    if not otm_puts_clean.empty:
        mean_iv_otm_put = otm_puts_clean['iv_close'].mean()
        logging.info(f"OTM Avg Skew: Calculated Mean OTM Put IV: {mean_iv_otm_put:.4f} from {len(otm_puts_clean)} options.")
    else:
        logging.warning("OTM Avg Skew: No valid OTM puts found to calculate average IV.")

    # --- 3. Find σ_ATM ---
    sigma_atm = np.nan
    atm_strike = np.nan
    try:
        valid_for_atm = dft_latest.dropna(subset=['k', 'iv_close'])
        valid_for_atm = valid_for_atm[valid_for_atm['iv_close'] > 1e-6]
        if not valid_for_atm.empty:
             atm_option_idx = abs(valid_for_atm['k'] - spot_price).idxmin()
             atm_option = valid_for_atm.loc[atm_option_idx]
             sigma_atm = atm_option['iv_close']
             atm_strike = atm_option['k']
             logging.info(f"OTM Avg Skew: Using σ_ATM from strike {atm_strike:.2f} (closest to spot {spot_price:.2f}): {sigma_atm:.4f}")
        else:
             logging.warning("OTM Avg Skew: No valid options found to determine ATM strike/IV.")
    except Exception as e:
        logging.error(f"OTM Avg Skew: Unexpected error finding ATM option IV: {e}", exc_info=True)

    # --- 4. Calculate Final Skew Value ---
    skew_value = np.nan
    calculation_possible = pd.notna(sigma_atm) and pd.notna(mean_iv_otm_call) and pd.notna(mean_iv_otm_put)

    if calculation_possible:
        if abs(sigma_atm) > 1e-9:
            iv_diff = mean_iv_otm_call - mean_iv_otm_put
            # Formula: (MeanCall - MeanPut) / ATM
            skew_value = iv_diff / sigma_atm
            logging.info(f"Calculated OTM Avg Skew: {skew_value:.4f} (MeanCallIV={mean_iv_otm_call:.4f}, MeanPutIV={mean_iv_otm_put:.4f}, σ_ATM={sigma_atm:.4f})")
        else:
            logging.warning("Cannot calculate OTM Avg Skew: σ_ATM is zero or near zero.")
    else:
        logging.warning(f"Cannot calculate OTM Avg Skew because one or more IV components are missing: "
                   f"MeanCallIV={mean_iv_otm_call}, MeanPutIV={mean_iv_otm_put}, σ_ATM={sigma_atm}")

    # --- 5. Display Results & Interpretation ---
    st.metric("Skew (Avg OTM C-P / ATM)", f"{skew_value:.4f}" if pd.notna(skew_value) else "N/A")

    if pd.notna(skew_value):
        threshold = 0.05 # Example threshold for significance
        if skew_value > threshold:
             st.info("Interpretation: Positive Skew - Average OTM calls are relatively more expensive than average OTM puts (vs ATM level). Suggests higher demand for upside.")
        elif skew_value < -threshold:
             st.info("Interpretation: Negative Skew - Average OTM puts are relatively more expensive than average OTM calls (vs ATM level). Suggests higher demand for downside protection.")
        else:
             st.info("Interpretation: Near-Zero Skew - Average OTM calls and puts have similar relative pricing (vs ATM level).")
    else:
         st.info("Interpretation: Skew could not be calculated. Check required IV values below.")

    # Display the intermediate IVs used
    with st.expander("Show IVs used for OTM Average Skew Calculation"):
        sigma_atm_disp = f"{sigma_atm:.4f}" if pd.notna(sigma_atm) else 'N/A'
        atm_strike_disp = f"{atm_strike:.0f}" if pd.notna(atm_strike) else 'N/A'
        mean_call_disp = f"{mean_iv_otm_call:.4f}" if pd.notna(mean_iv_otm_call) else 'N/A - No valid OTM Calls'
        mean_put_disp = f"{mean_iv_otm_put:.4f}" if pd.notna(mean_iv_otm_put) else 'N/A - No valid OTM Puts'

        st.caption(f"σ_ATM (Strike ~{atm_strike_disp}): {sigma_atm_disp}")
        st.caption(f"Mean(σ_otm_call): {mean_call_disp}")
        st.caption(f"Mean(σ_otm_put): {mean_put_disp}")

# Adapted Term Structure Plotting
# --- Helper to get ATM IV for one expiry from snapshot ---
def get_atm_iv_from_snapshot(df_paradex_options, expiry_dt, spot_price):
    """Gets ATM IV for a specific expiry from the snapshot DataFrame."""
    if df_paradex_options.empty or pd.isna(expiry_dt) or pd.isna(spot_price):
        return np.nan

    # Filter for the specific expiry
    expiry_options = df_paradex_options[df_paradex_options['expiry_date_utc'] == expiry_dt].copy()
    if expiry_options.empty:
        logging.warning(f"get_atm_iv_from_snapshot: No options found for expiry {expiry_dt} in snapshot.")
        return np.nan

    # Ensure numeric types for k and iv_close before proceeding
    expiry_options['k'] = pd.to_numeric(expiry_options['k'], errors='coerce')
    expiry_options['iv_close'] = pd.to_numeric(expiry_options['iv_close'], errors='coerce')

    # Clean and sort data
    expiry_options = expiry_options.dropna(subset=['k', 'iv_close']).sort_values('k')
    expiry_options = expiry_options[expiry_options['iv_close'] > 1e-6] # Filter non-positive IVs

    if len(expiry_options) < 1:
        logging.warning(f"get_atm_iv_from_snapshot: No valid options remain for expiry {expiry_dt} after cleaning.")
        return np.nan # Need at least one option

    strikes = expiry_options['k'].values
    ivs = expiry_options['iv_close'].values

    # Try interpolation first if possible
    if len(strikes) >= 2:
        try:
            # Use linear interpolation, don't extrapolate if spot is outside strike range
            interp_func = interp1d(strikes, ivs, kind='linear', bounds_error=False, fill_value=np.nan)
            atm_iv_interp = float(interp_func(spot_price))
            # Return interpolated value only if it's valid (not NaN and positive)
            if pd.notna(atm_iv_interp) and atm_iv_interp > 0:
                logging.debug(f"Interpolated ATM IV for {expiry_dt} from snapshot: {atm_iv_interp:.4f}")
                return atm_iv_interp
            else:
                logging.debug(f"Snapshot ATM IV interpolation returned invalid value ({atm_iv_interp}) for {expiry_dt}. Falling back to closest.")
        except Exception as e_interp:
            logging.warning(f"Snapshot ATM IV interpolation failed for {expiry_dt}: {e_interp}. Falling back.")
            # Fall through to closest strike if interpolation fails

    # Fallback: Find closest strike IV (use if only 1 strike or interpolation failed/invalid)
    try:
        closest_idx = np.argmin(np.abs(strikes - spot_price))
        atm_iv_closest = ivs[closest_idx]
        if pd.notna(atm_iv_closest) and atm_iv_closest > 0:
             logging.debug(f"Using closest strike ATM IV for {expiry_dt} from snapshot: {atm_iv_closest:.4f} (Strike: {strikes[closest_idx]:.0f})")
             return atm_iv_closest
        else:
            logging.warning(f"Closest strike ATM IV for {expiry_dt} is invalid ({atm_iv_closest}).")
            return np.nan
    except Exception as e_closest:
         logging.warning(f"Error finding closest strike IV for {expiry_dt} from snapshot: {e_closest}")
         return np.nan
# --- Main Term Structure Plot function (using snapshot data) ---
def plot_iv_term_structure_snapshot(df_paradex_options, spot_price, coin): # Added coin
    # ... (code remains the same as provided previously, ADDED coin to title) ...
    st.subheader("ATM IV Term Structure (Latest Snapshot)")
    if df_paradex_options.empty or pd.isna(spot_price): st.warning("Cannot plot snapshot term structure."); return
    now_utc = dt.datetime.now(dt.timezone.utc); expiries = sorted([d for d in df_paradex_options['expiry_date_utc'].dropna().unique() if pd.to_datetime(d) > now_utc])
    if not expiries: st.warning("No future expiries found in snapshot data."); return
    term_structure_points = []
    with st.spinner(f"Calculating Snapshot Term Structure ({len(expiries)} expiries)..."):
        for expiry_dt in expiries:
            atm_iv = get_atm_iv_from_snapshot(df_paradex_options, expiry_dt, spot_price)
            if pd.notna(atm_iv): T_term = max(1e-6, (expiry_dt - now_utc).total_seconds() / (365 * 24 * 3600)); term_structure_points.append({'Expiry_Date': expiry_dt, 'T_years': T_term, 'ATM_IV': atm_iv})
    if not term_structure_points: st.warning("Could not calculate any valid ATM IV points for term structure."); return
    term_structure_df = pd.DataFrame(term_structure_points); term_structure_df['Forward_IV'] = np.nan
    if len(term_structure_df) >= 2:
        try: # Calculate Fwd IV
            tsdf_plot=term_structure_df.sort_values('T_years').reset_index(drop=True); tsdf_plot['T1']=tsdf_plot['T_years'].shift(1); tsdf_plot['IV1']=tsdf_plot['ATM_IV'].shift(1); tsdf_plot['VarT1']=tsdf_plot['IV1']**2*tsdf_plot['T1']; tsdf_plot['VarT2']=tsdf_plot['ATM_IV']**2*tsdf_plot['T_years']; tsdf_plot['DeltaT']=tsdf_plot['T_years']-tsdf_plot['T1']; tsdf_plot['FVar']=np.where((tsdf_plot['DeltaT'].notna())&(tsdf_plot['DeltaT']>1e-9),(tsdf_plot['VarT2']-tsdf_plot['VarT1'])/tsdf_plot['DeltaT'],np.nan); tsdf_plot['FVar']=tsdf_plot['FVar'].clip(lower=0); fiv_series_plot=np.sqrt(tsdf_plot['FVar']); fiv_series_plot.index=tsdf_plot.index; term_structure_df['Forward_IV']=term_structure_df.index.map(fiv_series_plot)
        except Exception as e_fiv_plot: logging.error(f"Snapshot Fwd IV Plot Calc Err: {e_fiv_plot}")
    # --- Call the underlying plotting function ---
    plot_iv_term_structure_all_expiries(term_structure_df, spot_price, coin) # Pass coin

def plot_iv_term_structure_all_expiries(term_structure_data, spot_price, coin): # Added coin
    """Plots the IV term structure (ATM IV vs. Expiry Date) and Forward IV."""
    # NOTE: This function remains mostly the same as before, just ensure it takes the 'coin' parameter for the title.
    # ... (plotting code as defined previously, ensure title uses 'coin') ...
    st.subheader("ATM & Forward IV Term Structure") # Keep subheader separate
    if term_structure_data.empty: st.warning("Term structure data empty."); return
    required_cols = ['Expiry_Date', 'ATM_IV']
    if not all(c in term_structure_data.columns for c in required_cols): st.warning("TS data missing required columns."); return
    df_plot = term_structure_data.dropna(subset=['Expiry_Date', 'ATM_IV']).sort_values('Expiry_Date')
    if df_plot.empty: st.warning("No valid TS points after dropna."); return
    df_plot['ATM_IV_Plot'] = df_plot['ATM_IV'] * 100
    if 'Forward_IV' in df_plot.columns and df_plot['Forward_IV'].notna().any(): df_plot['Forward_IV_Plot'] = df_plot['Forward_IV'] * 100
    else: df_plot['Forward_IV_Plot'] = np.nan
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['Expiry_Date'], y=df_plot['ATM_IV_Plot'], mode='lines+markers', name='ATM IV', line=dict(color='black', width=2), marker=dict(size=7, color='black')))
        if 'Forward_IV_Plot' in df_plot.columns:
            df_fwd_plot = df_plot.dropna(subset=['Forward_IV_Plot'])
            if not df_fwd_plot.empty: fig.add_trace(go.Scatter(x=df_fwd_plot['Expiry_Date'], y=df_fwd_plot['Forward_IV_Plot'], mode='lines+markers', name='ATM Fwd IV', line=dict(color='grey', width=2, dash='dash'), marker=dict(size=7, color='grey')))
        fig.update_layout(title=f"{coin} ATM Volatility Term Structure (Spot: ${spot_price:,.0f})", xaxis_title=None, yaxis_title=None, yaxis_ticksuffix='%', height=400, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), hovermode="x unified", yaxis_gridcolor='rgba(211,211,211,0.5)', xaxis_gridcolor='rgba(211,211,211,0.5)', plot_bgcolor='white', yaxis_range=[max(0, df_plot[['ATM_IV_Plot', 'Forward_IV_Plot']].min(skipna=True).min(skipna=True) - 5) if not df_plot[['ATM_IV_Plot', 'Forward_IV_Plot']].min(skipna=True).isnull().all() else 0, (df_plot[['ATM_IV_Plot', 'Forward_IV_Plot']].max(skipna=True).max(skipna=True) + 5) if not df_plot[['ATM_IV_Plot', 'Forward_IV_Plot']].max(skipna=True).isnull().all() else 100])
        fig.update_xaxes(tickformat="%b-%y", gridcolor='rgba(211,211,211,0.5)', showline=True, linewidth=1, linecolor='lightgrey', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error plotting IV Term Structure: {e}"); logging.error(f"Error plotting IV Term Structure: {e}", exc_info=True)

# Market Memory Plots (Keep)
def calculate_btc_annualized_volatility_daily(df_daily):
    """Calculate annualized volatility for BTC using daily data over the last 30 days."""
    if df_daily.empty or 'close' not in df_daily.columns or len(df_daily) < 2:
        return np.nan
    df_daily = df_daily.dropna(subset=["close"]).copy()
    if len(df_daily) < 2:
        return np.nan
    df_daily["log_return"] = np.log(df_daily["close"] / df_daily["close"].shift(1))
    last_30 = df_daily["log_return"].dropna().tail(30)
    if len(last_30) < 2: # Need at least 2 returns for std dev
        return np.nan
    return last_30.std(ddof=1) * np.sqrt(365) # Use sample std dev
def compute_realized_volatility_5min(df, annualize_days=365):
    """Compute realized volatility using 5-minute log returns."""
    if df.empty or 'close' not in df.columns or len(df) < 2:
        return 0.0
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df_valid = df.dropna(subset=['log_ret'])
    if df_valid.empty:
        return 0.0
    total_variance = df_valid['log_ret'].pow(2).sum()
    if total_variance <= 0:
        return 0.0
    N = len(df_valid)
    M = annualize_days * 24 * 12 # 5-min intervals per year
    annualization_factor = np.sqrt(M / N) if N > 0 else 0
    return np.sqrt(total_variance) * annualization_factor
def calculate_hurst_lo_modified(series, min_n=10, max_n=None, n_step=1, q_method='auto'):
    """
    Calculates the Hurst exponent using Lo's (1991) modified R/S analysis,
    which is more robust to short-term dependence.

    Args:
        series (np.ndarray or pd.Series or list): The time series data (e.g., log returns).
        min_n (int): Minimum sub-period length n for R/S calculation.
        max_n (int): Maximum sub-period length n. Defaults to len(series) // 2.
        n_step (int): Step size for iterating through n values (IGNORED if using geomspace).
        q_method (str or int): Method to determine lag q for autocovariance adjustment.
                               'auto': Use heuristic q = floor(1.1447 * n^(1/3)), capped at n-1.
                                int: Use a fixed non-negative integer value for q.

    Returns:
        tuple: (hurst_exponent, results_df)
               hurst_exponent (float): The estimated Hurst exponent H. Returns np.nan on failure.
               results_df (pd.DataFrame): DataFrame with 'interval' (n) and 'rs_mean' (Avg(R/S)_q)
                                          used for the log-log regression. Empty on failure.
    """
    if isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values

    # Drop NaNs from the input series ONCE at the beginning
    series = series[~np.isnan(series)]
    N = len(series)

    if max_n is None:
        max_n = N // 2
    # Ensure n range is valid
    max_n = min(max_n, N - 1) # n must be less than N
    min_n = max(2, min_n)      # n must be at least 2

    if N < 20 or min_n >= max_n:
        logging.warning(f"Series too short (N={N}) or invalid n range ({min_n}-{max_n}) for reliable Hurst (Lo) calculation.")
        return np.nan, pd.DataFrame()

    # Generate intervals (logarithmic spacing is generally preferred)
    ns = np.unique(np.geomspace(min_n, max_n, num=20, dtype=int)) # Logarithmic spacing
    ns = [n_val for n_val in ns if n_val >= min_n] # Ensure min_n is respected

    if not ns:
         logging.warning("No valid window sizes 'n' found in the specified range.")
         return np.nan, pd.DataFrame()

    rs_values = []
    valid_ns = []

    logging.info(f"Calculating Lo's Mod R/S for n in {ns}...")
    for n in ns:
        # Determine q for this n
        q = 0 # Default to 0
        if isinstance(q_method, int):
            q = max(0, min(q_method, n - 1)) # Use fixed q, ensure 0 <= q < n
        elif q_method == 'auto':
            # Heuristic for q based on n
            if n > 10: # Heuristic may be unstable for very small n
                 # Andrews (1991) / Newey-West (1994) style heuristic often used:
                 q = int(np.floor(1.1447 * ((n)**(1/3)))) # Example based on n^(1/3) rule
                 q = max(0, min(q, n - 1)) # Ensure 0 <= q < n
        else:
            logging.error(f"Invalid q_method: {q_method}. Use 'auto' or an integer. Defaulting q=0.")
            q = 0

        rs_chunk = []
        num_chunks = N // n
        if num_chunks == 0: continue # Should not happen if max_n < N-1

        # --- Process Sub-periods (Chunks) ---
        for i in range(num_chunks):
            start = i * n
            end = start + n
            chunk = series[start:end]

            # Calculate Range (R) - Same as classic R/S
            mean = np.mean(chunk)

            # Check for constant chunk BEFORE calculating R and S
            if np.allclose(chunk, mean):
                continue # Skip constant chunks (R=0, S_q=0)

            mean_adjusted = chunk - mean
            cum_dev = np.cumsum(mean_adjusted)
            cum_dev_with_zero = np.insert(cum_dev, 0, 0.0)
            R = np.ptp(cum_dev_with_zero)

            if pd.isna(R) or R < 0: # Range must be non-negative
                 logging.debug(f"Skipping chunk at n={n}, i={i}: Invalid Range R={R}")
                 continue

            # Calculate Lo's Modified Standard Deviation (S_q)
            modified_var = calculate_lo_modified_variance(chunk, q)

            if pd.isna(modified_var):
                 logging.debug(f"Skipping chunk at n={n}, i={i}: Failed modified variance calc.")
                 continue

            # --- Calculate R/S_q Ratio ---
            if modified_var > 1e-12: # Check variance > 0 before sqrt
                S_q = np.sqrt(modified_var)
                rs = R / S_q
                if not pd.isna(rs) and rs >= 0: # R/S ratio must be non-negative and finite
                    rs_chunk.append(rs)
                else:
                    logging.debug(f"Skipping chunk at n={n}, i={i}: Invalid R/S_q ratio (R={R}, S_q={S_q}, ratio={rs})")
            # else: # Log if modified_var is too small
                # logging.debug(f"Skipping chunk at n={n}, i={i}: Modified Variance near zero ({modified_var})")


        # --- Average R/S_q for this n ---
        if rs_chunk: # Only if we got valid R/S ratios for this n
            avg_rs_q = np.mean(rs_chunk)
            rs_values.append(avg_rs_q)
            valid_ns.append(n) # Store the n value for which we got a result

    # --- Log-Log Regression ---
    if len(valid_ns) < 3: # Need at least 3 points for a reliable regression
        logging.warning(f"Insufficient valid R/S points ({len(valid_ns)}) for Hurst (Lo) regression.")
        return np.nan, pd.DataFrame()

    results_df = pd.DataFrame({'interval': valid_ns, 'rs_mean': rs_values})
    # Ensure no NaN/inf values made it into the final rs_values
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(results_df) < 3:
        logging.warning(f"Insufficient valid R/S points ({len(results_df)}) after final dropna for Hurst (Lo) regression.")
        return np.nan, pd.DataFrame()

    log_intervals = np.log(results_df['interval'])
    log_rs = np.log(results_df['rs_mean'])

    try:
        # Fit log-log plot: log(R/S_q) = log(c) + H * log(n)
        hurst, intercept, r_value, p_value, std_err = linregress(log_intervals, log_rs)

        # Basic check on Hurst value reasonableness
        if not (0 < hurst < 1):
            logging.warning(f"Calculated Hurst (Lo) exponent ({hurst:.4f}) is outside the typical (0, 1) range.")

        logging.info(f"Lo's Modified Hurst Exponent Calculation Successful. H = {hurst:.4f}")
        return hurst, results_df # Return H and the data used for plotting

    except (np.linalg.LinAlgError, ValueError) as e:
        logging.error(f"Linear algebra/Value error during Hurst (Lo) polyfit: {e}")
        return np.nan, results_df
    except Exception as e:
        logging.error(f"Unexpected error during Hurst (Lo) polyfit: {e}")
        return np.nan, results_df
def plot_hurst_exponent(hurst_val, hurst_data_df): # Signature unchanged
    """Plots the log-log R/S data and the fitted Hurst line."""
    if pd.isna(hurst_val) or hurst_data_df.empty:
        st.write("Hurst Exponent could not be calculated or plotted.")
        return

    # Check if DataFrame contains necessary columns
    if not all(c in hurst_data_df.columns for c in ['interval', 'rs_mean']):
         st.error("Hurst data DataFrame missing 'interval' or 'rs_mean' columns.")
         logging.error("Hurst data DF structure incorrect for plotting.")
         return

    # Prepare data, handling potential log(0) or log(negative) if rs_mean was somehow <= 0
    plot_data = hurst_data_df[hurst_data_df['rs_mean'] > 1e-12].copy() # Filter for positive R/S
    if plot_data.empty:
         st.warning("No plot data remaining after filtering non-positive R/S values.")
         return
    if len(plot_data) < 3:
         st.warning(f"Only {len(plot_data)} points remaining for Hurst plot after filtering.")
         # Optionally, decide if you still want to plot just the points

    log_intervals = np.log(plot_data['interval'])
    log_rs = np.log(plot_data['rs_mean'])

    # Re-calculate fitted line for plotting using only valid plot_data points
    fitted_rs = None
    label_h = hurst_val # Use the originally calculated H for the label
    try:
        # Perform regression again just for plotting consistency on the filtered data
        # Check again if enough points remain for regression
        if len(plot_data) >= 3:
            hurst_plot, intercept_plot, r_val_plot, p_val_plot, stderr_plot = linregress(log_intervals, log_rs)
            fitted_rs = intercept_plot + hurst_plot * log_intervals # Use intercept from regression
            # You might choose to update label_h = hurst_plot here if you prefer the plot fit H
            # label_h = hurst_plot
        else:
            logging.warning("Not enough points to calculate fitted line for Hurst plot.")

    except Exception as e: # Handle potential errors
         logging.warning(f"Could not generate fitted line for Hurst plot: {e}")


    fig_hurst = go.Figure()
    # Plot Markers
    fig_hurst.add_trace(go.Scatter(
        x=log_intervals, y=log_rs, mode='markers', name='Log(Avg R/S<SUB>q</SUB>) vs Log(n)', marker=dict(color='blue')
    ))
    # Plot Fitted Line if available
    if fitted_rs is not None:
        fig_hurst.add_trace(go.Scatter(
            x=log_intervals, y=fitted_rs, mode='lines', name=f'Fit (H={label_h:.3f})', line=dict(color='red', dash='dash')
        ))

    # Update Layout (Using SUB tags for subscript q)
    fig_hurst.update_layout(title="Hurst Exponent Log-Log Plot (Lo's Modified R/S)",
                            xaxis_title="Log(Time Interval n)", yaxis_title="Log(Mean R/S<SUB>q</SUB>)",
                            height=400, width=800)
    st.plotly_chart(fig_hurst, use_container_width=True)

    # Interpret Hurst value (using the originally calculated hurst_val)
    if 0.55 < hurst_val <= 1.0: regime = "Trending / Persistent (H > 0.5)" # Added upper bound check
    elif 0.0 <= hurst_val < 0.45: regime = "Mean-Reverting / Anti-Persistent (H < 0.5)" # Added lower bound check
    elif 0.45 <= hurst_val <= 0.55: regime = "Random Walk / Brownian Motion (H ≈ 0.5)"
    else: regime = f"Unusual Value (H={hurst_val:.3f})" # Handle cases outside [0, 1]

    st.write(f"**Hurst Exponent (Lo's Mod. R/S, Daily Log Returns): {hurst_val:.3f}**") # Clarified label
    st.write(f"**Implied Market Regime:** {regime}")
def calculate_and_display_autocorrelation(daily_log_returns, windows=[7, 15, 30], threshold=0.05):
    """
    Calculates lag-1 autocorrelation and displays the implied regime symbol
    (≈, —, ↑, ↓) for different recent windows, with a textual description
    below each symbol using the system's default text color and centered alignment.

    Symbol Logic & Descriptions:
    - ≈ (Mean Reverting): Autocorr < -threshold
    - — (Random Walk): |Autocorr| <= threshold
    - ↑ (Persistent Up Trend): Autocorr > threshold AND Avg Return > 0
    - ↓ (Persistent Down Trend): Autocorr > threshold AND Avg Return <= 0
    - ? (Unknown): Insufficient data or calculation issue.
    - ! (Error): Calculation error.

    Args:
        daily_log_returns (pd.Series): Series of daily log returns, ideally indexed by date.
        windows (list, optional): List of lookback periods (in days). Defaults to [7, 15, 30].
        threshold (float, optional): Threshold around zero to classify autocorrelation
                                     as significant (positive or negative). Defaults to 0.05.
    """
    st.subheader("Implied Market Regime (Daily Autocorrelation)")

    # Input Validation
    if not isinstance(daily_log_returns, pd.Series):
        st.error("Autocorrelation Error: Input must be a pandas Series.")
        logging.error("Autocorrelation input was not a pandas Series.")
        return
    if daily_log_returns.empty:
        st.warning("Cannot determine regime: Daily returns data is empty.")
        return
    cleaned_returns = daily_log_returns.dropna()
    if cleaned_returns.empty:
        st.warning("Cannot determine regime: No valid daily returns after dropping NaNs.")
        return

    # Symbol to Description Mapping
    regime_map = {
        "≈": "Mean Reverting",
        "—": "Random Walk",
        "↑": "Persistent Up",
        "↓": "Persistent Down",
        "?": "Unknown",
        "!": "Error"
    }

    cols = st.columns(len(windows))

    for i, window in enumerate(windows):
        with cols[i]:
            regime_symbol = "?" # Default symbol

            # Calculation Logic (remains the same)
            if len(cleaned_returns) < window:
                logging.warning(f"Autocorrelation: Not enough total valid data ({len(cleaned_returns)}) for {window}-day window.")
            elif len(cleaned_returns.tail(window)) < 2:
                logging.warning(f"Autocorrelation: Less than 2 valid points in the last {window} days.")
            else:
                subset = cleaned_returns.tail(window)
                try:
                    autocorr_val = subset.autocorr(lag=1)

                    if pd.isna(autocorr_val):
                         logging.warning(f"Autocorrelation: Calculation returned NaN for {window}-day window.")
                    elif autocorr_val < -threshold:
                        regime_symbol = "≈" # Mean Reverting
                    elif autocorr_val > threshold:
                        # Trending / Persistent Case
                        avg_return = subset.mean()
                        if pd.isna(avg_return):
                             pass # Keep '?'
                        elif avg_return > 0:
                             regime_symbol = "↑" # Persistent Up Trend
                        else: # avg_return <= 0
                             regime_symbol = "↓" # Persistent Down Trend
                    else:
                        # Neutral / Random Walk Case
                        regime_symbol = "—"

                except Exception as e:
                    logging.error(f"Error calculating autocorrelation for {window}-day window: {e}")
                    regime_symbol = "!" # Symbol for error

            # Get description text
            description = regime_map.get(regime_symbol, "Unknown")

            # --- DISPLAY: Symbol + Description Below (System Color) ---
            # Display window length (small, grey, centered)
            st.markdown(f"<p style='text-align: center; font-size: small; color: grey; margin-bottom: -10px;'>{window}-Day</p>", unsafe_allow_html=True)
            # Display symbol (large, centered)
            st.markdown(f"<h1 style='text-align: center; margin-bottom: -10px;'>{regime_symbol}</h1>", unsafe_allow_html=True)
            # Display description (small, centered, *using default text color*)
            st.markdown(f"<p style='text-align: center; font-size: small; margin-top: 0px;'>{description}</p>", unsafe_allow_html=True) # <-- REMOVED 'color: white;'
import scipy.interpolate # Add this import at the top

# Net Greeks Calculation (Keep)
def calculate_net_vega(df_latest_snap):
    # ... (code remains the same, takes snapshot df) ...
    required_cols = ['vega', 'open_interest']
    if df_latest_snap.empty or not all(c in df_latest_snap.columns for c in required_cols): return np.nan
    df_clean = df_latest_snap.dropna(subset=required_cols).copy(); df_clean['vega'] = pd.to_numeric(df_clean['vega'], errors='coerce'); df_clean['open_interest'] = pd.to_numeric(df_clean['open_interest'], errors='coerce'); df_clean = df_clean.dropna(subset=required_cols)
    if df_clean.empty: return np.nan; net_vega = (df_clean['vega'] * df_clean['open_interest']).sum(); return net_vega

def calculate_net_charm(df_latest_snap):
    # ... (code remains the same, takes snapshot df) ...
    required_cols = ['charm', 'open_interest']
    if df_latest_snap.empty or not all(c in df_latest_snap.columns for c in required_cols): return np.nan
    df_clean = df_latest_snap.dropna(subset=required_cols).copy(); df_clean['charm'] = pd.to_numeric(df_clean['charm'], errors='coerce'); df_clean['open_interest'] = pd.to_numeric(df_clean['open_interest'], errors='coerce'); df_clean = df_clean.dropna(subset=required_cols)
    if df_clean.empty: return np.nan; net_charm = (df_clean['charm'] * df_clean['open_interest']).sum(); return net_charm if np.isfinite(net_charm) else np.nan

def calculate_net_vanna(df_latest_snap):
    # ... (code remains the same, takes snapshot df) ...
    required_cols = ['vanna', 'open_interest']
    if df_latest_snap.empty or not all(c in df_latest_snap.columns for c in required_cols): return np.nan
    df_clean = df_latest_snap.dropna(subset=required_cols).copy(); df_clean['vanna'] = pd.to_numeric(df_clean['vanna'], errors='coerce'); df_clean['open_interest'] = pd.to_numeric(df_clean['open_interest'], errors='coerce'); df_clean = df_clean.dropna(subset=required_cols)
    if df_clean.empty: return np.nan; net_vanna = (df_clean['vanna'] * df_clean['open_interest']).sum(); return net_vanna

# --- Safe Plot Helper ---
def safe_plot(plot_func, *args, **kwargs):
    plot_name = getattr(plot_func, '__name__', 'N/A')
    try:
        if callable(plot_func):
            logging.info(f"Attempting to plot: {plot_name}")
            plot_func(*args, **kwargs)
            logging.info(f"Successfully plotted: {plot_name}")
        else:
            logging.error(f"Plot function not callable: {plot_name}")
            st.error(f"Plot function invalid: {plot_name}")
    except Exception as e:
        st.error(f"Plot error in '{plot_name}'. Check logs.")
        logging.error(f"Plot error in {plot_name}", exc_info=True)

# --- Main App ---
def main():
    # login() # Uncomment if you want login functionality

    # --- Initialize ---
    df_paradex_options = pd.DataFrame()
    dft_latest = pd.DataFrame()
    ticker_list = []
    df_krak_5m = pd.DataFrame()
    df_krak_daily = pd.DataFrame()

    st.session_state.snapshot_time = dt.datetime.now(dt.timezone.utc)
    logging.info(f"Paradex Dashboard execution started: {st.session_state.snapshot_time}")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH"], key='selected_coin').upper()

    # --- Fetch Paradex Snapshot ---
    with st.spinner("Fetching Paradex markets snapshot..."):
        paradex_snapshot_data = fetch_paradex_markets_snapshot()

    if not paradex_snapshot_data:
        st.error("Failed to fetch market data from Paradex Testnet API. Cannot proceed.")
        st.stop()

    # --- Process Snapshot Data ---
    with st.spinner("Processing options data..."):
        df_paradex_options = filter_and_process_paradex_options(paradex_snapshot_data)

    if df_paradex_options.empty:
        st.error("No valid options data found in the Paradex snapshot.")
        st.stop()

    # Filter for selected coin
    df_paradex_options_coin = df_paradex_options[df_paradex_options['underlying_asset'] == coin].copy()
    if df_paradex_options_coin.empty:
        st.error(f"No {coin} options found in the snapshot.")
        st.stop()

    # --- Get Expiries and Select ---
    expiry_options = get_valid_expiration_options_from_snapshot(df_paradex_options_coin)
    if not expiry_options:
        st.error(f"No valid future expiration dates found for {coin}. Cannot proceed.")
        st.stop()

    # Simple default: first expiry
    default_expiry_index = 0
    selected_expiry = st.sidebar.selectbox(
        "Choose Expiration Date",
        options=expiry_options,
        format_func=lambda dt_obj: dt_obj.strftime("%d %b %Y (%H:%M UTC)"),
        index=default_expiry_index,
        key="paradex_expiry_select"
    )
    e_str_display = selected_expiry.strftime('%d%b%y') # For display

    # --- Filter Data for Selected Expiry ---
    dft_latest = df_paradex_options_coin[df_paradex_options_coin['expiry_date_utc'] == selected_expiry].copy()
    if dft_latest.empty:
        st.error(f"No {coin} options found for selected expiry {e_str_display} in the snapshot.")
        st.stop()

    # --- Fetch Spot Data (for context) ---
    with st.spinner(f"Fetching Kraken {coin} spot data..."):
        df_krak_5m = fetch_kraken_data(coin=coin, days=1)
        df_krak_daily = fetch_kraken_data_daily(days=365, coin=coin)

    if df_krak_5m.empty: st.error(f"Failed to fetch Kraken {coin} 5min data."); st.stop()
    if df_krak_daily.empty: st.error(f"Failed to fetch Kraken {coin} daily data."); st.stop()

    spot_price = df_krak_5m["close"].iloc[-1]

    # --- Calculate T_years for selected expiry ---
    T_years = max(0.00001, (selected_expiry - st.session_state.snapshot_time).total_seconds() / (365 * 24 * 3600))
    days_to_expiry_val = max(0, (selected_expiry - st.session_state.snapshot_time).days) # Ensure non-negative
    st.sidebar.write(f"Days to Expiry (Selected): {days_to_expiry_val}")

    # --- Calculate/Verify Greeks in dft_latest ---
    greeks_to_calc = []
    greek_cols = ['delta', 'gamma', 'vega', 'charm', 'vanna', 'theta']
    for greek in greek_cols:
        if greek not in dft_latest.columns or dft_latest[greek].isna().all():
            greeks_to_calc.append(greek)
            if greek not in dft_latest.columns: dft_latest[greek] = np.nan

    if greeks_to_calc:
        logging.warning(f"Greeks {greeks_to_calc} missing/NaN. Calculating...")
        with st.spinner(f"Calculating missing Greeks ({', '.join(greeks_to_calc)})..."):
            snapshot_time = st.session_state.snapshot_time
            for index, row in dft_latest.iterrows():
                # Re-calculate T for each row (though it should be ~constant for snapshot)
                # T_row = max(1e-6, (row['expiry_date_utc'] - snapshot_time).total_seconds() / (365 * 24 * 3600))
                # Using pre-calculated T_years for efficiency as it's a snapshot
                if 'delta' in greeks_to_calc: dft_latest.loc[index, 'delta'] = compute_delta(row, spot_price, snapshot_time)
                if 'gamma' in greeks_to_calc: dft_latest.loc[index, 'gamma'] = compute_gamma(row, spot_price, snapshot_time)
                if 'vega' in greeks_to_calc: dft_latest.loc[index, 'vega'] = compute_vega(row, spot_price, snapshot_time)
                if 'charm' in greeks_to_calc: dft_latest.loc[index, 'charm'] = compute_charm(row, spot_price, snapshot_time)
                if 'vanna' in greeks_to_calc: dft_latest.loc[index, 'vanna'] = compute_vanna(row, spot_price, snapshot_time)
                # Add theta if needed
            for greek in greeks_to_calc:
                 if greek in dft_latest.columns: dft_latest[greek] = dft_latest[greek].astype('float32')
            logging.info(f"Finished calculating missing Greeks: {greeks_to_calc}")
    else:
        logging.info("Greeks appear present in Paradex snapshot data.")
        for greek in greek_cols: # Ensure type
            if greek in dft_latest.columns: dft_latest[greek] = pd.to_numeric(dft_latest[greek], errors='coerce').astype('float32')

    # Calculate GEX
    if 'gamma' in dft_latest.columns and 'open_interest' in dft_latest.columns:
        dft_latest['gex'] = dft_latest.apply(lambda row: compute_gex(row, spot_price, row['open_interest']), axis=1).astype('float32')
    else: dft_latest['gex'] = np.nan

    # --- Build Ticker List (from processed dft_latest) ---
    ticker_list = build_paradex_ticker_list(dft_latest)

    # --- Strike Filter ---
    dev_opt = st.sidebar.select_slider(
        "Filter Strike Range (Snapshot Views)",
        options=["±0.5σ (IV Est.)", "±1.0σ (IV Est.)", "±1.5σ (IV Est.)", "±2.0σ (IV Est.)", "All"],
        value="±2.0σ (IV Est.)", key="paradex_strike_filter"
    )
    multiplier = float('inf') if dev_opt == "All" else float(dev_opt.split('σ')[0].replace('±',''))

    dft_latest_filtered = dft_latest.copy()
    if multiplier != float('inf'):
        atm_iv_filter = get_atm_iv_from_snapshot(dft_latest, selected_expiry, spot_price)
        iv_for_filter = atm_iv_filter if pd.notna(atm_iv_filter) else dft_latest['iv_close'].mean()
        if pd.isna(iv_for_filter) or iv_for_filter <=0 : iv_for_filter = 0.5
        try:
            exp_term = iv_for_filter * np.sqrt(T_years) * multiplier
            lo_bound = spot_price * np.exp(-exp_term); hi_bound = spot_price * np.exp(exp_term)
            logging.info(f"Filtering snapshot strikes between: {lo_bound:.2f} and {hi_bound:.2f}")
            dft_latest_filtered = dft_latest[(dft_latest['k'] >= lo_bound) & (dft_latest['k'] <= hi_bound)].copy()
        except Exception as e_filter: st.error(f"Error applying filter: {e_filter}"); logging.error("Error applying snapshot filter", exc_info=True)

    df_ticker_list_filtered = pd.DataFrame(build_paradex_ticker_list(dft_latest_filtered))


    # --- Display Header ---
    st.title(f"Paradex {coin} Options Snapshot")
    st.header(f"Analysis for Expiry: {selected_expiry.strftime('%d %b %Y')} | Spot: ${spot_price:,.2f}")
    st.markdown(f"*Snapshot Time (UTC): {st.session_state.snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}*")

    # --- Dashboard Layout ---

    # --- Key Metrics & Market Memory ---
    st.markdown("---")
    st.header("Key Metrics & Market Memory")
    col_metrics, col_memory = st.columns([0.4, 0.6])
    with col_metrics:
        st.subheader("Volatility Snapshot")
        rv_30d = calculate_btc_annualized_volatility_daily(df_krak_daily)
        rv_7d_5m = compute_realized_volatility_5min(df_krak_5m.tail(288*7))
        latest_iv_mean_selected = dft_latest['iv_close'].mean() if not dft_latest.empty else np.nan
        st.metric("Realized Vol (30d Daily)", f"{rv_30d:.2%}" if pd.notna(rv_30d) else "N/A")
        st.metric("Realized Vol (7d 5-min)", f"{rv_7d_5m:.2%}" if pd.notna(rv_7d_5m) and rv_7d_5m > 0 else "N/A")
        st.metric(f"Mean IV ({e_str_display})", f"{latest_iv_mean_selected:.2%}" if pd.notna(latest_iv_mean_selected) else "N/A")
        st.markdown("---")
        # Term Structure (use full processed options df for this)
        safe_plot(plot_iv_term_structure_snapshot, df_paradex_options_coin, spot_price, coin)

    with col_memory:
        st.subheader("Hurst Exponent (Market Memory - Lo's Mod. R/S)")
        daily_log_returns = np.log(df_krak_daily['close'] / df_krak_daily['close'].shift(1))
        hurst_val, hurst_data_df = calculate_hurst_lo_modified(daily_log_returns, q_method='auto')
        safe_plot(plot_hurst_exponent, hurst_val, hurst_data_df)
        st.markdown("---")
        st.subheader("Autocorrelation (Market Memory)")
        safe_plot(calculate_and_display_autocorrelation, daily_log_returns, windows=[7, 15, 30])


    # --- Market Maker Positioning ---
    st.markdown("---")
    st.header(f"Market Maker Positioning (Snapshot - {e_str_display})")
    st.caption("Shows aggregate risk exposure assuming MMs sold the open interest.")
    net_delta_latest = (dft_latest['delta'] * dft_latest['open_interest']).sum() if 'delta' in dft_latest.columns else np.nan
    calls_gex = dft_latest.loc[dft_latest['option_type'] == 'C', 'gex'].sum(skipna=True) if 'gex' in dft_latest.columns else 0
    puts_gex = dft_latest.loc[dft_latest['option_type'] == 'P', 'gex'].sum(skipna=True) if 'gex' in dft_latest.columns else 0
    net_gex_latest = calls_gex - puts_gex if pd.notna(calls_gex) and pd.notna(puts_gex) else np.nan
    net_vega_latest = calculate_net_vega(dft_latest)
    net_vanna_latest = calculate_net_vanna(dft_latest)
    net_charm_latest = calculate_net_charm(dft_latest)

    col_mm1, col_mm2, col_mm3, col_mm4, col_mm5 = st.columns(5)
    with col_mm1: st.metric("Net Delta", f"{net_delta_latest:.2f}" if pd.notna(net_delta_latest) else "N/A"); # Add captions if needed
    with col_mm2: st.metric("Net GEX", f"{net_gex_latest:,.0f}" if pd.notna(net_gex_latest) else "N/A"); # Add captions if needed
    with col_mm3: st.metric("Net Vega", f"{net_vega_latest:,.0f}" if pd.notna(net_vega_latest) else "N/A"); # Add captions if needed
    with col_mm4: st.metric("Net Vanna", f"{net_vanna_latest:,.0f}" if pd.notna(net_vanna_latest) else "N/A"); # Add captions if needed
    with col_mm5: st.metric("Net Charm", f"{net_charm_latest:,.2f}" if pd.notna(net_charm_latest) else "N/A"); # Add captions if needed
    st.info("**Disclaimer:** Simplified view based on aggregate OI.")

    # --- Open Interest & Greeks ---
    st.markdown("---")
    st.header(f"Open Interest & Greeks (Snapshot - {e_str_display})")
    st.caption("Use sidebar slider to filter strike range for bar charts.")

    # Plots using UNFILTERED ticker_list
    safe_plot(plot_oi_by_strike, ticker_list, spot_price)
    safe_plot(plot_open_interest_delta, ticker_list, spot_price)
    safe_plot(plot_delta_balance, ticker_list, spot_price)
    # Plots using FILTERED data
    safe_plot(plot_premium_by_strike, dft_latest_filtered, spot_price)
    safe_plot(plot_net_delta, df_ticker_list_filtered, spot_price)
    safe_plot(plot_gex_by_strike, dft_latest_filtered)
    safe_plot(plot_net_gex, dft_latest_filtered, spot_price)

    # --- Max Pain ---
    st.markdown("---")
    st.header(f"Max Pain (Snapshot - {e_str_display})")
    safe_plot(plot_call_put_max_pain, dft_latest, spot_price)
    safe_plot(plot_call_put_max_pain_in_range, dft_latest, spot_price)

    # --- Volatility Smile & Skew ---
    st.markdown("---")
    st.header(f"Volatility & Skew (Snapshot - {e_str_display})")
    safe_plot(plot_volatility_smile, dft_latest, spot_price)
    safe_plot(compute_and_display_otm_average_skew, dft_latest, spot_price)

    # --- Raw Data Table ---
    st.markdown("---")
    st.header(f"Raw Data Table (Snapshot - {e_str_display})")
    with st.expander("Show Processed Option Data for Selected Expiry"):
        if not dft_latest.empty:
            display_cols = [c for c in ['instrument_name', 'k', 'option_type', 'mark_price_close', 'iv_close', 'open_interest', 'delta', 'gamma', 'vega', 'theta', 'charm', 'vanna', 'gex'] if c in dft_latest.columns]
            st.dataframe(dft_latest[display_cols].round(4), use_container_width=True)
        else: st.info("No data for selected expiry.")

    # --- Final Cleanup ---
    logging.info("Performing final cleanup...")
    # ... (Remove variables specific to historical plots/sims if desired) ...
    gc.collect()
    logging.info(f"Paradex Snapshot Dashboard rendering complete for {coin} {e_str_display}.")


# --- Entry Point ---
if __name__ == "__main__":
    main()
