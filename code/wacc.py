"""Estimate the client's WACC.

Run from the repo root:   python code/wacc.py
Reads:  inputs/assumptions.csv
Writes: outputs/wacc.csv (one row; the columns are listed in README.md)

Owner: the developer. Fill in the three TODO functions (your agent can help),
then run it and check the numbers make sense before you open a pull request.
"""
import pandas as pd
import yfinance as yf
import math

CLIENT = 'TGT'
MKT = 'SPY'
YEARS = 5


def get_returns(ticker, mkt=MKT, years=YEARS):
    """Daily returns for the client and the market over the last `years` years.
    Returns a DataFrame with two columns, named ticker and mkt, and no missing values."""
    prices = yf.download(
        [ticker, mkt], period=f'{years}y', interval='1d',
        auto_adjust=True, progress=False,
    )['Close'][[ticker, mkt]]
    assert not prices.empty, 'no prices downloaded'
    assert prices.notna().sum().min() > 1, 'missing price history'
    rets = prices.pct_change(fill_method=None).dropna()
    assert not rets.empty and not rets.isna().any().any(), 'missing returns'
    assert rets.map(math.isfinite).all().all(), 'non-finite returns'
    return rets


def estimate_beta(rets, ticker, mkt=MKT):
    """Equity beta: cov(stock, market) / var(market)."""
    assert len(rets) > 1, 'need at least two return observations'
    assert rets[[ticker, mkt]].map(math.isfinite).all().all(), 'invalid returns'
    market_variance = rets[mkt].var()
    assert market_variance > 0, 'market variance must be positive'
    return rets[ticker].cov(rets[mkt]) / market_variance


def compute_wacc(beta, a):
    """a is a dict of assumptions: rf, mrp, rd, tax, w_e, w_d.
    Returns a dict with cost_of_equity and wacc (and the inputs), matching the README's columns."""
    inputs = {key: float(a[key]) for key in ('rf', 'mrp', 'rd', 'tax', 'w_e', 'w_d')}
    assert math.isfinite(beta) and all(map(math.isfinite, inputs.values())), 'invalid inputs'
    assert abs(inputs['w_e'] + inputs['w_d'] - 1) < 1e-6, 'weights must sum to 1'
    assert all(0 <= inputs[key] <= 1 for key in ('tax', 'w_e', 'w_d')), 'invalid tax or weights'
    cost_of_equity = inputs['rf'] + beta * inputs['mrp']
    wacc = inputs['w_e'] * cost_of_equity + inputs['w_d'] * inputs['rd'] * (1 - inputs['tax'])
    assert 0 < wacc < 0.25, 'WACC outside 0-25%: check the inputs'
    return {**inputs, 'cost_of_equity': cost_of_equity, 'wacc': wacc}


if __name__ == '__main__':
    a = pd.read_csv('inputs/assumptions.csv', index_col='item')['value'].to_dict()
    assert abs(a['w_e'] + a['w_d'] - 1) < 1e-6, 'weights must sum to 1'

    rets = get_returns(CLIENT)
    assert len(rets) > 1000, 'expected about 5 years of daily returns'

    beta = estimate_beta(rets, CLIENT)
    out = compute_wacc(beta, a)
    out = {'ticker': CLIENT, 'asof': rets.index.max().date(), 'beta': beta, **out}

    assert 0 < out['wacc'] < 0.25, 'WACC outside 0-25%: check the inputs'
    cols = ['ticker', 'asof', 'beta', 'rf', 'mrp', 'cost_of_equity', 'rd', 'tax', 'w_e', 'w_d', 'wacc']
    pd.DataFrame([out])[cols].to_csv('outputs/wacc.csv', index=False)
    print(pd.DataFrame([out])[cols].T)
