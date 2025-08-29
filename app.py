import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def fetch_ohlcv(ticker: str, start: dt.date, end: dt.date, auto_adjust: bool = True) -> pd.DataFrame:
    end_inclusive = end + dt.timedelta(days=1)
    df = yf.download(
        ticker,
        start=str(start),
        end=str(end_inclusive),
        auto_adjust=auto_adjust,
        group_by="column",
    ).dropna()

    # Normalize columns from yfinance across versions/symbol modes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in col if str(level) != ""]).strip("_")
            for col in df.columns
        ]

    def pick_column(field: str) -> str:
        if field in df.columns:
            return field
        candidates = [name for name in df.columns if name.endswith(f"_{field}") or name.startswith(f"{field}_")]
        if not candidates:
            raise KeyError(f"Could not find column for field '{field}' in: {list(df.columns)}")
        return candidates[0]

    open_col = pick_column("Open")
    high_col = pick_column("High")
    low_col = pick_column("Low")
    close_col = pick_column("Close")
    volume_col = pick_column("Volume")

    df = df[[open_col, high_col, low_col, close_col, volume_col]].rename(
        columns={
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
            volume_col: "Volume",
        }
    )
    return df


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_strategy(
    df: pd.DataFrame,
    window: int,
    rule_mode: str,
    threshold_pct: float,
    position_mode: str,
    use_trailing_sl: bool,
    sl_pct: float,
    use_take_profit: bool,
    tp_pct: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = df["Close"]
    ma = compute_sma(close, window)

    # Thresholds for hysteresis
    up_level = ma * (1.0 + threshold_pct)
    down_level = ma * (1.0 - threshold_pct)

    # Compute desired state from rules (without stops/targets)
    # 1 = long, 0 = flat, -1 = short
    if rule_mode == "Regime (above/below)":
        if position_mode == "Long only":
            desired_state = (close > up_level).astype(int)
        elif position_mode == "Short only":
            desired_state = -(close < down_level).astype(int)
        else:  # Long/Short
            desired_state = pd.Series(0, index=df.index)
            desired_state[close > up_level] = 1
            desired_state[close < down_level] = -1
    else:
        # True hysteresis with explicit edge-cross detection
        desired_state = pd.Series(0, index=df.index, dtype=int)
        prev_state = 0  # 0 = flat, 1 = long, -1 = short
        for i in range(len(df)):
            c = close.iloc[i]
            ul = up_level.iloc[i]
            dl = down_level.iloc[i]

            if prev_state == 0:
                # Enter only on crossing outside the bands
                if position_mode in ("Long only", "Long/Short") and c > ul:
                    prev_state = 1
                elif position_mode in ("Short only", "Long/Short") and c < dl:
                    prev_state = -1
            elif prev_state == 1:
                # While long, exit only when crossing below the lower band
                if c < dl:
                    if position_mode == "Long/Short" and c < dl:
                        prev_state = -1  # flip to short in L/S
                    else:
                        prev_state = 0   # go flat in Long-only
            elif prev_state == -1:
                # While short, exit only when crossing above the upper band
                if c > ul:
                    if position_mode == "Long/Short" and c > ul:
                        prev_state = 1   # flip to long in L/S
                    else:
                        prev_state = 0   # go flat in Short-only

            # Enforce caps for Long-only / Short-only after transitions
            if position_mode == "Long only" and prev_state == -1:
                prev_state = 0
            if position_mode == "Short only" and prev_state == 1:
                prev_state = 0

            desired_state.iloc[i] = prev_state

    # Trade-aware execution with optional trailing stop-loss and take-profit.
    # Position values: 1 = long, 0 = flat, -1 = short
    position_list = []
    current_pos = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = np.nan
    peak_price = np.nan  # highest close since entry for trailing stop
    trough_price = np.nan  # lowest close since entry for trailing stop (shorts)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    desired = desired_state.values

    for i in range(len(df)):
        if current_pos == 0:
            # Consider entry on next bar open; approximate with today's close by using shift later in returns
            if desired[i] == 1:  # Go long
                current_pos = 1
                entry_price = closes[i]
                peak_price = entry_price
                trough_price = entry_price
            elif desired[i] == -1:  # Go short
                current_pos = -1
                entry_price = closes[i]
                peak_price = entry_price
                trough_price = entry_price
        elif current_pos == 1:  # Currently long
            # Update peak for trailing stop
            if closes[i] > peak_price:
                peak_price = closes[i]

            exit_due_to_rule = (desired[i] != 1)
            exit_due_to_tp = False
            exit_due_to_sl = False

            if use_take_profit and tp_pct > 0:
                tp_level = entry_price * (1.0 + tp_pct)
                if highs[i] >= tp_level:
                    exit_due_to_tp = True

            if use_trailing_sl and sl_pct > 0:
                # Trailing stop moves up with peak
                trail_level = peak_price * (1.0 - sl_pct)
                if lows[i] <= trail_level:
                    exit_due_to_sl = True

            if exit_due_to_tp or exit_due_to_sl or exit_due_to_rule:
                current_pos = 0
                entry_price = np.nan
                peak_price = np.nan
                trough_price = np.nan
        elif current_pos == -1:  # Currently short
            # Update trough for trailing stop (shorts)
            if closes[i] < trough_price:
                trough_price = closes[i]

            exit_due_to_rule = (desired[i] != -1)
            exit_due_to_tp = False
            exit_due_to_sl = False

            if use_take_profit and tp_pct > 0:
                tp_level = entry_price * (1.0 - tp_pct)  # For shorts, TP is below entry
                if lows[i] <= tp_level:
                    exit_due_to_tp = True

            if use_trailing_sl and sl_pct > 0:
                # Trailing stop moves down with trough for shorts
                trail_level = trough_price * (1.0 + sl_pct)
                if highs[i] >= trail_level:
                    exit_due_to_sl = True

            if exit_due_to_tp or exit_due_to_sl or exit_due_to_rule:
                current_pos = 0
                entry_price = np.nan
                peak_price = np.nan
                trough_price = np.nan

        position_list.append(current_pos)

    position = pd.Series(position_list, index=df.index, dtype=int)
    # Apply one-bar delay to avoid lookahead (use yesterday's position)
    exec_position = position.shift(1).fillna(0).astype(int)

    mkt_ret = close.pct_change().fillna(0)
    strat_ret = exec_position * mkt_ret  # Long positions gain on up moves, shorts gain on down moves
    return ma, mkt_ret, strat_ret, exec_position


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def sharpe_annualized(daily_ret: pd.Series) -> float:
    mean = daily_ret.mean()
    std = daily_ret.std(ddof=1)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(252))


@st.cache_data(show_spinner=False)
def fetch_ohlcv_cached(ticker: str, start: dt.date, end: dt.date, auto_adjust: bool) -> pd.DataFrame:
    return fetch_ohlcv(ticker, start, end, auto_adjust)

@st.cache_data(show_spinner=False)
def run_strategy_cached(df: pd.DataFrame,
                        window: int,
                        rule_mode: str,
                        threshold_pct: float,
                        position_mode: str,
                        use_trailing_sl: bool,
                        sl_pct: float,
                        use_take_profit: bool,
                        tp_pct: float):
    return compute_strategy(df, window, rule_mode, threshold_pct, position_mode, use_trailing_sl, sl_pct, use_take_profit, tp_pct)


st.set_page_config(page_title="SMA Strategy Playground", layout="wide")

# Custom CSS to widen sidebar columns by 5%
st.markdown("""
    <style>
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d2b4a;
    }
    
    /* Widen sidebar columns by 5% */
    .css-1d391kg {
        width: 105% !important;
    }
    .css-1d391kg > div {
        width: 105% !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Simple Moving Average (SMA) Strategy")

with st.sidebar:
    st.header("Parameters")
    
    with st.expander("Data Settings", expanded=True):
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        today = dt.date.today()
        default_start = today.replace(year=today.year - 2)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=today)
        sma_window = st.number_input("SMA window", min_value=2, max_value=1000, value=200, step=1)
        auto_adjust = st.checkbox("Auto-adjust prices (dividends/splits)", value=True)
    
    with st.expander("Entry/Exit Rules", expanded=True):
        position_mode = st.selectbox(
            "Position mode",
            options=["Long only", "Short only", "Long/Short"],
            index=0,
            help="Long only: Buy when signal is true. Short only: Sell when signal is false. Long/Short: Both directions."
        )
        rule_mode = st.selectbox(
            "Rule mode",
            options=["Regime (above/below)", "Cross edges with hysteresis"],
            index=1,
            help=(
                "Regime: Long when price > SMA(1+thr), flat otherwise.\n"
                "Cross edges: Enter only when crossing above (1+thr) level and exit only when crossing below (1-thr)."
            ),
        )
        threshold_pct = st.number_input(
            "Threshold (percent)", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
            help="Add buffer around SMA to avoid whipsaws. 0 = no buffer."
        ) / 100.0
    
    with st.expander("Stops / Targets", expanded=False):
        col_sl, col_tp = st.columns(2)
        with col_sl:
            use_trailing_sl = st.checkbox("Use trailing stop-loss", value=False)
            sl_pct = st.number_input("SL %", min_value=0.0, max_value=50.0, value=5.0, step=0.5) / 100.0
        with col_tp:
            use_take_profit = st.checkbox("Use take-profit", value=False)
            tp_pct = st.number_input("TP %", min_value=0.0, max_value=200.0, value=10.0, step=0.5) / 100.0

# Main area with Run Backtest button on the right
col1, col2, col3, col4, col5 = st.columns(5)
with col5:
    run_btn = st.button("Run Backtest", use_container_width=True)

if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    try:
        with st.spinner("Fetching data..."):
            df = fetch_ohlcv_cached(ticker, start_date, end_date, auto_adjust)
            if df.empty:
                st.error("No data returned for the selected range. Try different dates or ticker.")
                st.stop()
            if sma_window >= len(df):
                st.error(f"SMA window ({sma_window}) must be smaller than number of rows ({len(df)}).")
                st.stop()

        with st.spinner("Computing strategy..."):
            ma, ret_mkt, ret_strat, position = run_strategy_cached(
                df, sma_window, rule_mode, threshold_pct, position_mode,
                use_trailing_sl, sl_pct, use_take_profit, tp_pct
            )
            equity_strat = (1 + ret_strat).cumprod()
            equity_bh = (1 + ret_mkt).cumprod()

        final_equity = float(equity_strat.iloc[-1]) if not equity_strat.empty else 1.0
        total_return = final_equity - 1.0
        mdd = max_drawdown(equity_strat) if not equity_strat.empty else 0.0
        sharpe = sharpe_annualized(ret_strat) if not ret_strat.empty else 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{total_return:.2%}")
        col2.metric("Final Equity", f"{final_equity:.4f}")
        col3.metric("Sharpe (ann.)", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{mdd:.2%}")
        col5.metric("Rows", f"{len(df)}")

        st.subheader("Price vs SMA")
        chart_df = pd.DataFrame({"Close": df["Close"], f"SMA_{sma_window}": ma})
        st.line_chart(chart_df)

        st.subheader("Equity Curves")
        equity_df = pd.DataFrame({"SMA Strategy": equity_strat, "Buy & Hold": equity_bh})
        st.line_chart(equity_df)

        st.subheader("Position (0=flat, 1=long, -1=short)")
        st.line_chart(position)

        st.subheader("Daily Return Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Market:")
            st.write(ret_mkt.describe())
        with c2:
            st.write("Strategy:")
            st.write(ret_strat.describe())

        st.subheader("Data Preview")
        st.write(f"Rows: {len(df)}")
        st.dataframe(df.head())
        st.dataframe(df.tail())

    except Exception as exc:
        st.error(f"Error: {exc}")
        st.stop()
else:
    st.info("Set parameters and click 'Run Backtest' to begin.")
