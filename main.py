import pandas as pd
import numpy as np
import yfinance as yf

# Use adjusted OHLC so price & SMA live on the same (post-split) scale.
df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2021-01-01",
    auto_adjust=True,
    group_by='column'
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

OPEN = pick_column('Open')
HIGH = pick_column('High')
LOW = pick_column('Low')
CLOSE = pick_column('Close')
VOLUME = pick_column('Volume')

# Restrict to core columns with normalized names for consistency
df = df[[OPEN, HIGH, LOW, CLOSE, VOLUME]].rename(columns={
    OPEN: 'Open', HIGH: 'High', LOW: 'Low', CLOSE: 'Close', VOLUME: 'Volume'
})

# Sanity Check for DataFrame

print('Rows:', len(df))
print(df.head(5))
print(df.tail(5))

print("\nScale check (raw close sample):")
print(df[['Close']].tail(3))


def sma(series: pd.Series, n: int) -> pd.Series:

    return series.rolling(window=n, min_periods=n).mean()

N = 200
ma = sma(df['Close'], N)

print("\nLatest Close vs SMA:")
print(pd.DataFrame({
    'Close': df['Close'].tail(5),
    f'SMA_{N}': ma.tail(5)
}))
'''
print(ma.head(5))
print(ma.head(205))
print(ma.tail(5))
'''
# Position signal: compute with today's data, apply next day to avoid lookahead
raw_signal = (df['Close'] > ma)
position = raw_signal.shift(1).fillna(False).astype(int)

print("\nPosition value counts (0=flat,1=long):")
print(position.value_counts().sort_index())
print("Position preview:")
print(position.tail(10))

# Market daily returns
ret_mkt = df['Close'].pct_change().fillna(0)

# Strategy daily returns (use yesterday's position already encoded in `position`)
ret_strat = position * ret_mkt

# Buy & Hold equity for comparison
ret_bh = ret_mkt
equity_bh = (1 + ret_bh).cumprod()
'''
print("\nMarket daily return stats:", ret_mkt.describe())
print("Strategy daily return stats:", ret_strat.describe())
'''
equity = (1 + ret_strat).cumprod()
print("Final equity (start=1.0):", float(equity.iloc[-1]))

def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())  # negative number

def sharpe_annualized(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    # rf_annual: annual risk-free rate; use 0.0 to keep it simple
    mean = daily_ret.mean()
    std = daily_ret.std(ddof=0)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(252))

final_equity = float(equity.iloc[-1])
total_return = final_equity - 1.0
mdd = max_drawdown(equity)
sharpe = sharpe_annualized(ret_strat)

'''
print("\n=== RESULTS ===")
print(f"Total Return: {total_return:.2%}")
print(f"Final Equity: {final_equity:.4f} (start=1.0000)")
print(f"Sharpe (ann.): {sharpe:.2f}")
print(f"Max Drawdown: {mdd:.2%}")   
print(f"Buy & Hold Final Equity: {float(equity_bh.iloc[-1]):.4f}")
'''
# Graphs removed - use app.py for interactive visualization